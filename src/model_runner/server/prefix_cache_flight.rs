use crate::model_runner::PrefixCacheCoordinationPolicy;
use anyhow::bail;
use anyhow::ensure;
use anyhow::Context;
use anyhow::Result;
use std::collections::HashMap;
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
    state: FlightState,
    follower_request_ids: VecDeque<u64>,
}

#[derive(Clone, Copy)]
enum FlightState {
    Active { leader_request_id: u64 },
    Draining,
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
        self.state = FlightState::Active {
            leader_request_id: next_leader_request_id,
        };
        Some(next_leader_request_id)
    }

    /// Transitions the flight after its leader completes and returns the followers to make runnable.
    fn begin_draining(&mut self) -> Vec<u64> {
        self.state = FlightState::Draining;
        self.follower_request_ids.iter().copied().collect()
    }
}

/// Coordinates prefix-cache population by allowing one request in each flight to execute.
///
/// Flight membership is independent of the matching strategy. The initial exact-input index can
/// later be replaced by prefix-aware matching without changing request or flight bookkeeping.
pub(super) struct PrefixCacheFlightCoordinator {
    next_flight_id: FlightId,
    flights: HashMap<FlightId, Flight>,
    request_flight_ids: HashMap<u64, FlightId>,
    exact_input_index: HashMap<Arc<[u32]>, FlightId>,
}

impl PrefixCacheFlightCoordinator {
    /// Creates a coordinator when the configured policy requires flight coordination.
    pub(super) fn from_policy(policy: PrefixCacheCoordinationPolicy) -> Option<Self> {
        match policy {
            PrefixCacheCoordinationPolicy::Independent => None,
            PrefixCacheCoordinationPolicy::ExactInputSingleFlight => Some(Self::new()),
        }
    }

    fn new() -> Self {
        Self {
            next_flight_id: FlightId(0),
            flights: HashMap::new(),
            request_flight_ids: HashMap::new(),
            exact_input_index: HashMap::new(),
        }
    }

    /// Adds a request and returns whether it should execute or wait for the current leader.
    pub(super) fn add_request(
        &mut self,
        request_id: u64,
        input_token_ids: &[u32],
    ) -> Result<FlightAdmission> {
        ensure!(
            !self.request_flight_ids.contains_key(&request_id),
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

    /// Removes a terminated request and returns any requests that have become runnable.
    ///
    /// A completed leader releases every follower. An aborted leader promotes the oldest follower.
    /// Followers can only be removed as aborted before release because they have not started
    /// execution. Released followers remain tracked until termination and return no runnable IDs.
    pub(super) fn remove_request(
        &mut self,
        request_id: u64,
        request_termination: RequestTermination,
    ) -> Result<Vec<u64>> {
        let flight_id = self
            .request_flight_ids
            .get(&request_id)
            .copied()
            .with_context(|| {
                format!("prefix-cache flight request {request_id} is not registered")
            })?;
        let flight_state = self.flight_unchecked(flight_id).state;
        let runnable_request_ids = match (flight_state, request_termination) {
            (FlightState::Active { leader_request_id }, RequestTermination::Completed)
                if leader_request_id == request_id =>
            {
                self.complete_leader(flight_id, request_id)
            }
            (FlightState::Active { leader_request_id }, RequestTermination::Aborted)
                if leader_request_id == request_id =>
            {
                let flight = self.flight_mut_unchecked(flight_id);
                if let Some(promoted_request_id) = flight.elect_next_leader() {
                    self.request_flight_ids.remove(&request_id);
                    vec![promoted_request_id]
                } else {
                    self.remove_flight(flight_id);
                    Vec::new()
                }
            }
            (FlightState::Active { .. }, RequestTermination::Aborted) => {
                let flight = self.flight_mut_unchecked(flight_id);
                flight.remove_follower(request_id);
                self.request_flight_ids.remove(&request_id);
                Vec::new()
            }
            (FlightState::Active { .. }, RequestTermination::Completed) => {
                bail!("prefix-cache flight follower {request_id} should not complete before leader")
            }
            (FlightState::Draining, _) => {
                self.request_flight_ids.remove(&request_id);
                let flight = self.flight_mut_unchecked(flight_id);
                flight.remove_follower(request_id);
                if flight.follower_request_ids.is_empty() {
                    self.remove_flight(flight_id);
                }
                Vec::new()
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
                state: FlightState::Active { leader_request_id },
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
        if let FlightState::Active { leader_request_id } = flight.state {
            self.exact_input_index.remove(&flight.input_token_ids);
            self.request_flight_ids.remove(&leader_request_id);
        }
        for follower_request_id in &flight.follower_request_ids {
            self.request_flight_ids.remove(follower_request_id);
        }
        flight
    }

    fn complete_leader(&mut self, flight_id: FlightId, leader_request_id: u64) -> Vec<u64> {
        let flight = self.flight_mut_unchecked(flight_id);
        let runnable_request_ids = flight.begin_draining();
        let input_token_ids = Arc::clone(&flight.input_token_ids);
        // Stop accepting followers: an identical request creates a new flight while this one drains.
        self.exact_input_index.remove(&input_token_ids);
        self.request_flight_ids.remove(&leader_request_id);
        if runnable_request_ids.is_empty() {
            self.remove_flight(flight_id);
        }
        runnable_request_ids
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
        assert!(PrefixCacheFlightCoordinator::from_policy(
            PrefixCacheCoordinationPolicy::ExactInputSingleFlight
        )
        .is_some());
    }

    #[test]
    fn adds_leaders_and_ordered_followers_by_exact_input() -> Result<()> {
        let mut coordinator = PrefixCacheFlightCoordinator::new();

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
    fn completed_leader_releases_followers_and_drains_flight() -> Result<()> {
        let mut coordinator = PrefixCacheFlightCoordinator::new();
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
    fn aborted_leader_promotes_oldest_follower() -> Result<()> {
        let mut coordinator = PrefixCacheFlightCoordinator::new();
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
        let mut coordinator = PrefixCacheFlightCoordinator::new();
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
        let mut coordinator = PrefixCacheFlightCoordinator::new();
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
