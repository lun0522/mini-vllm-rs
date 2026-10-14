use crate::model_runner::PrefixCacheCoordinationPolicy;
use anyhow::bail;
use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;
use std::sync::Arc;

#[derive(Debug, Eq, PartialEq)]
pub(super) enum FlightAdmission {
    Leader,
    Follower,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum RequestTermination {
    Completed,
    Aborted,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum FlightReleaseCondition {
    RequestCompletion,
    PrefixPublication,
}

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
struct FlightId(u64);

impl FlightId {
    fn next(&mut self) -> Result<Self> {
        let current = *self;
        self.0 = self
            .0
            .checked_add(1)
            .context("prefix-cache flight ID overflow")?;
        Ok(current)
    }
}

struct Flight {
    input_token_ids: Arc<[u32]>,
    leader_request_id: u64,
    follower_request_ids: VecDeque<u64>,
}

impl Flight {
    fn add_follower(&mut self, request_id: u64) {
        self.follower_request_ids.push_back(request_id);
    }

    fn remove_follower(&mut self, request_id: u64) {
        let follower_position = self
            .follower_request_ids
            .iter()
            .position(|follower_request_id| *follower_request_id == request_id)
            .expect("a registered non-leader request must be a follower");
        self.follower_request_ids.remove(follower_position);
    }

    fn elect_next_leader(&mut self) -> Option<u64> {
        let next_leader_request_id = self.follower_request_ids.pop_front()?;
        self.leader_request_id = next_leader_request_id;
        Some(next_leader_request_id)
    }
}

/// Coordinates prefix-cache population by allowing one request in each flight to execute.
///
/// Flight membership is independent of the matching strategy. The initial exact-input index can
/// later be replaced by prefix-aware matching without changing request or flight bookkeeping.
pub(super) struct PrefixCacheFlightCoordinator {
    release_condition: FlightReleaseCondition,
    next_flight_id: FlightId,
    flights: HashMap<FlightId, Flight>,
    request_flight_ids: HashMap<u64, FlightId>,
    exact_input_index: HashMap<Arc<[u32]>, FlightId>,
    released_request_ids: HashSet<u64>,
}

impl PrefixCacheFlightCoordinator {
    /// Creates a coordinator when the configured policy requires flight coordination.
    pub(super) fn from_policy(policy: PrefixCacheCoordinationPolicy) -> Option<Self> {
        match policy {
            PrefixCacheCoordinationPolicy::Independent => None,
            PrefixCacheCoordinationPolicy::ExactInputSingleFlightUntilRequestCompletion => {
                Some(Self::new(FlightReleaseCondition::RequestCompletion))
            }
            PrefixCacheCoordinationPolicy::ExactInputSingleFlightUntilPrefixPublished => {
                Some(Self::new(FlightReleaseCondition::PrefixPublication))
            }
        }
    }

    fn new(release_condition: FlightReleaseCondition) -> Self {
        Self {
            release_condition,
            next_flight_id: FlightId(0),
            flights: HashMap::new(),
            request_flight_ids: HashMap::new(),
            exact_input_index: HashMap::new(),
            released_request_ids: HashSet::new(),
        }
    }

    /// Returns whether the request leads a flight that releases after prefix publication.
    pub(super) fn requires_prefix_publication(&self, request_id: u64) -> bool {
        if self.release_condition != FlightReleaseCondition::PrefixPublication {
            return false;
        }
        let Some(flight_id) = self.request_flight_ids.get(&request_id) else {
            return false;
        };
        self.flight_unchecked(*flight_id).leader_request_id == request_id
    }

    /// Adds a request and returns whether it should execute or wait for the current leader.
    pub(super) fn add_request(
        &mut self,
        request_id: u64,
        input_token_ids: &[u32],
    ) -> Result<FlightAdmission> {
        ensure!(
            !self.request_flight_ids.contains_key(&request_id)
                && !self.released_request_ids.contains(&request_id),
            "prefix-cache flight request {request_id} is already registered"
        );
        if let Some(flight_id) = self.exact_input_index.get(input_token_ids).copied() {
            let flight = self.flight_mut_unchecked(flight_id);
            flight.add_follower(request_id);
            self.request_flight_ids.insert(request_id, flight_id);
            Ok(FlightAdmission::Follower)
        } else {
            self.add_flight(request_id, input_token_ids)?;
            Ok(FlightAdmission::Leader)
        }
    }

    /// Releases a leader's followers after successful prefix publication and returns their IDs.
    /// `request_id` must identify the flight's leader request.
    pub(super) fn release_after_prefix_publication(&mut self, request_id: u64) -> Result<Vec<u64>> {
        if self.release_condition != FlightReleaseCondition::PrefixPublication {
            return Ok(Vec::new());
        }
        let flight_id = self
            .request_flight_ids
            .get(&request_id)
            .copied()
            .with_context(|| {
                format!("prefix-cache flight request {request_id} is not registered")
            })?;
        ensure!(
            self.flight_unchecked(flight_id).leader_request_id == request_id,
            "prefix-cache flight follower {request_id} cannot publish before its leader"
        );
        Ok(self.release_followers(flight_id, /* is_leader_still_active */ true))
    }

    /// Removes a terminated request and returns any requests that have become runnable.
    ///
    /// A completed leader releases every follower. An aborted leader promotes the oldest follower.
    /// Followers can only be removed as aborted before release because they have not started
    /// execution. Released requests remain tracked until termination and return no runnable IDs.
    pub(super) fn remove_request(
        &mut self,
        request_id: u64,
        request_termination: RequestTermination,
    ) -> Result<Vec<u64>> {
        if self.released_request_ids.remove(&request_id) {
            return Ok(Vec::new());
        }

        let flight_id = self
            .request_flight_ids
            .get(&request_id)
            .copied()
            .with_context(|| {
                format!("prefix-cache flight request {request_id} is not registered")
            })?;
        let is_leader = self.flight_unchecked(flight_id).leader_request_id == request_id;
        let runnable_request_ids = match (is_leader, request_termination) {
            (/* is_leader */ true, RequestTermination::Completed) => {
                self.release_followers(flight_id, /* is_leader_still_active */ false)
            }
            (/* is_leader */ true, RequestTermination::Aborted) => {
                let flight = self.flight_mut_unchecked(flight_id);
                if let Some(promoted_request_id) = flight.elect_next_leader() {
                    self.request_flight_ids.remove(&request_id);
                    vec![promoted_request_id]
                } else {
                    self.remove_flight(flight_id);
                    Vec::new()
                }
            }
            (/* is_leader */ false, RequestTermination::Aborted) => {
                let flight = self.flight_mut_unchecked(flight_id);
                flight.remove_follower(request_id);
                self.request_flight_ids.remove(&request_id);
                Vec::new()
            }
            (/* is_leader */ false, RequestTermination::Completed) => {
                bail!("prefix-cache flight follower {request_id} should not complete before leader")
            }
        };
        Ok(runnable_request_ids)
    }

    fn flight_unchecked(&self, flight_id: FlightId) -> &Flight {
        self.flights
            .get(&flight_id)
            .expect("a tracked flight ID must reference an existing flight")
    }

    fn flight_mut_unchecked(&mut self, flight_id: FlightId) -> &mut Flight {
        self.flights
            .get_mut(&flight_id)
            .expect("a tracked flight ID must reference an existing flight")
    }

    fn add_flight(&mut self, leader_request_id: u64, input_token_ids: &[u32]) -> Result<()> {
        let flight_id = self.next_flight_id.next()?;
        let input_token_ids: Arc<[u32]> = input_token_ids.into();
        self.exact_input_index
            .insert(Arc::clone(&input_token_ids), flight_id);
        self.flights.insert(
            flight_id,
            Flight {
                input_token_ids,
                leader_request_id,
                follower_request_ids: VecDeque::new(),
            },
        );
        self.request_flight_ids.insert(leader_request_id, flight_id);
        Ok(())
    }

    fn remove_flight(&mut self, flight_id: FlightId) -> Flight {
        let flight = self
            .flights
            .remove(&flight_id)
            .expect("the flight must exist");
        self.exact_input_index.remove(&flight.input_token_ids);
        self.request_flight_ids.remove(&flight.leader_request_id);
        for follower_request_id in &flight.follower_request_ids {
            self.request_flight_ids.remove(follower_request_id);
        }
        flight
    }

    /// Removes the flight, marks its still-running members as released, and returns the follower
    /// request IDs that should become runnable.
    fn release_followers(&mut self, flight_id: FlightId, is_leader_still_active: bool) -> Vec<u64> {
        let flight = self.remove_flight(flight_id);
        if is_leader_still_active {
            self.released_request_ids.insert(flight.leader_request_id);
        }
        self.released_request_ids
            .extend(flight.follower_request_ids.iter().copied());
        flight.follower_request_ids.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn creates_a_coordinator_only_for_single_flight_policy() {
        assert!(PrefixCacheFlightCoordinator::from_policy(
            PrefixCacheCoordinationPolicy::Independent
        )
        .is_none());
        let completion_coordinator = PrefixCacheFlightCoordinator::from_policy(
            PrefixCacheCoordinationPolicy::ExactInputSingleFlightUntilRequestCompletion,
        )
        .unwrap();
        assert_eq!(
            completion_coordinator.release_condition,
            FlightReleaseCondition::RequestCompletion
        );
        let publication_coordinator = PrefixCacheFlightCoordinator::from_policy(
            PrefixCacheCoordinationPolicy::ExactInputSingleFlightUntilPrefixPublished,
        )
        .unwrap();
        assert_eq!(
            publication_coordinator.release_condition,
            FlightReleaseCondition::PrefixPublication
        );
    }

    #[test]
    fn adds_leaders_and_ordered_followers_by_exact_input() -> Result<()> {
        let mut coordinator =
            PrefixCacheFlightCoordinator::new(FlightReleaseCondition::RequestCompletion);

        assert_eq!(
            coordinator.add_request(1, &[10, 20])?,
            FlightAdmission::Leader
        );
        assert_eq!(
            coordinator.add_request(2, &[10, 20])?,
            FlightAdmission::Follower
        );
        assert_eq!(
            coordinator.add_request(3, &[10, 21])?,
            FlightAdmission::Leader
        );
        Ok(())
    }

    #[test]
    fn completed_leader_releases_followers_and_removes_flight() -> Result<()> {
        let mut coordinator =
            PrefixCacheFlightCoordinator::new(FlightReleaseCondition::RequestCompletion);
        coordinator.add_request(1, &[10, 20])?;
        coordinator.add_request(2, &[10, 20])?;
        coordinator.add_request(3, &[10, 20])?;

        assert_eq!(
            coordinator.remove_request(1, RequestTermination::Completed)?,
            vec![2, 3]
        );
        assert_eq!(
            coordinator.add_request(4, &[10, 20])?,
            FlightAdmission::Leader
        );
        assert!(coordinator
            .remove_request(2, RequestTermination::Completed)?
            .is_empty());
        assert!(coordinator
            .remove_request(3, RequestTermination::Aborted)?
            .is_empty());
        assert_eq!(
            coordinator.add_request(5, &[10, 20])?,
            FlightAdmission::Follower
        );
        Ok(())
    }

    #[test]
    fn published_prefix_releases_followers_while_tracking_the_leader() -> Result<()> {
        let mut coordinator =
            PrefixCacheFlightCoordinator::new(FlightReleaseCondition::PrefixPublication);
        coordinator.add_request(1, &[10, 20])?;
        coordinator.add_request(2, &[10, 20])?;
        coordinator.add_request(3, &[10, 20])?;

        assert_eq!(coordinator.release_after_prefix_publication(1)?, vec![2, 3]);
        assert_eq!(
            coordinator.add_request(4, &[10, 20])?,
            FlightAdmission::Leader
        );
        assert!(coordinator
            .remove_request(2, RequestTermination::Completed)?
            .is_empty());
        assert!(coordinator
            .remove_request(1, RequestTermination::Completed)?
            .is_empty());
        assert!(coordinator
            .remove_request(3, RequestTermination::Aborted)?
            .is_empty());
        assert_eq!(
            coordinator.add_request(5, &[10, 20])?,
            FlightAdmission::Follower
        );
        Ok(())
    }

    #[test]
    fn completion_release_policy_ignores_prefix_publication() -> Result<()> {
        let mut coordinator =
            PrefixCacheFlightCoordinator::new(FlightReleaseCondition::RequestCompletion);
        coordinator.add_request(1, &[10, 20])?;
        coordinator.add_request(2, &[10, 20])?;

        assert!(coordinator.release_after_prefix_publication(1)?.is_empty());
        assert_eq!(
            coordinator.remove_request(1, RequestTermination::Completed)?,
            vec![2]
        );
        Ok(())
    }

    #[test]
    fn aborted_leader_promotes_oldest_follower() -> Result<()> {
        let mut coordinator =
            PrefixCacheFlightCoordinator::new(FlightReleaseCondition::RequestCompletion);
        coordinator.add_request(1, &[10, 20])?;
        coordinator.add_request(2, &[10, 20])?;
        coordinator.add_request(3, &[10, 20])?;

        assert_eq!(
            coordinator.remove_request(1, RequestTermination::Aborted)?,
            vec![2]
        );
        assert_eq!(
            coordinator.remove_request(2, RequestTermination::Completed)?,
            vec![3]
        );
        coordinator.add_request(4, &[30, 40])?;
        assert_eq!(
            coordinator.remove_request(4, RequestTermination::Aborted)?,
            Vec::<u64>::new()
        );
        Ok(())
    }

    #[test]
    fn aborted_follower_preserves_the_flight() -> Result<()> {
        let mut coordinator =
            PrefixCacheFlightCoordinator::new(FlightReleaseCondition::RequestCompletion);
        coordinator.add_request(1, &[10, 20])?;
        coordinator.add_request(2, &[10, 20])?;
        coordinator.add_request(3, &[10, 20])?;

        assert_eq!(
            coordinator.remove_request(2, RequestTermination::Aborted)?,
            Vec::<u64>::new()
        );
        assert_eq!(
            coordinator.remove_request(1, RequestTermination::Completed)?,
            vec![3]
        );
        Ok(())
    }

    #[test]
    fn rejects_invalid_request_lifecycle() -> Result<()> {
        let mut coordinator =
            PrefixCacheFlightCoordinator::new(FlightReleaseCondition::RequestCompletion);
        coordinator.add_request(1, &[10, 20])?;
        coordinator.add_request(2, &[10, 20])?;

        assert_eq!(
            coordinator
                .add_request(1, &[30, 40])
                .unwrap_err()
                .to_string(),
            "prefix-cache flight request 1 is already registered"
        );
        assert_eq!(
            coordinator
                .remove_request(2, RequestTermination::Completed)
                .unwrap_err()
                .to_string(),
            "prefix-cache flight follower 2 should not complete before leader"
        );
        assert_eq!(
            coordinator
                .remove_request(3, RequestTermination::Aborted)
                .unwrap_err()
                .to_string(),
            "prefix-cache flight request 3 is not registered"
        );
        Ok(())
    }
}
