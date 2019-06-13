# Copyright (c) 2017, John Skinner
import unittest
import arvet.core.benchmark
import arvet.core.trial_comparison
import arvet_slam.trials.slam.tracking_state as tracking_state
import arvet_slam.benchmarks.tracking.tracking_comparison_benchmark as track_comp


class MockTrialResult:

    def __init__(self, tracking_states):
        self._tracking_states = tracking_states

    @property
    def identifier(self):
        return 'ThisIsAMockTrialResult'

    @property
    def tracking_states(self):
        return self._tracking_states

    @tracking_states.setter
    def tracking_states(self, tracking_states):
        self._tracking_states = tracking_states

    def get_tracking_states(self):
        return self.tracking_states


class TestTrackingComparisonBenchmark(unittest.TestCase):

    def test_benchmark_results_returns_a_benchmark_result(self):
        trial_result = MockTrialResult(tracking_states={
            1.3333: tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: tracking_state.TrackingState.OK
        })
        reference_trial_result = MockTrialResult(tracking_states={
            1.3333: tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: tracking_state.TrackingState.NOT_INITIALIZED
        })
        benchmark = track_comp.TrackingComparisonBenchmark()
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertIsInstance(result, arvet.core.trial_comparison.TrialComparisonResult)
        self.assertNotIsInstance(result, arvet.core.benchmark.FailedBenchmark)
        self.assertEqual(benchmark.identifier, result.benchmark)
        self.assertEqual(trial_result.identifier, result.trial_result)
        self.assertEqual(reference_trial_result.identifier, result.reference_trial_result)

    def test_benchmark_produces_expected_results(self):
        trial_result = MockTrialResult(tracking_states={
            1.3333: tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: tracking_state.TrackingState.OK,
            2: tracking_state.TrackingState.LOST,
            2.3333: tracking_state.TrackingState.NOT_INITIALIZED,
            2.6667: tracking_state.TrackingState.OK,
            3: tracking_state.TrackingState.LOST,
            3.3333: tracking_state.TrackingState.NOT_INITIALIZED,
            3.6667: tracking_state.TrackingState.OK,
            4: tracking_state.TrackingState.LOST
        })
        reference_trial_result = MockTrialResult(tracking_states={
            1.3333: tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: tracking_state.TrackingState.NOT_INITIALIZED,
            2: tracking_state.TrackingState.NOT_INITIALIZED,
            2.3333: tracking_state.TrackingState.OK,
            2.6667: tracking_state.TrackingState.OK,
            3: tracking_state.TrackingState.OK,
            3.3333: tracking_state.TrackingState.LOST,
            3.6667: tracking_state.TrackingState.LOST,
            4: tracking_state.TrackingState.LOST
        })
        benchmark = track_comp.TrackingComparisonBenchmark()
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertNotIsInstance(result, arvet.core.benchmark.FailedBenchmark)

        self.assertEqual(6, len(result.changes))
        self.assertEqual((tracking_state.TrackingState.NOT_INITIALIZED, tracking_state.TrackingState.OK),
                         result.changes[1.6667])
        self.assertEqual((tracking_state.TrackingState.NOT_INITIALIZED, tracking_state.TrackingState.LOST),
                         result.changes[2])
        self.assertEqual((tracking_state.TrackingState.OK, tracking_state.TrackingState.NOT_INITIALIZED),
                         result.changes[2.3333])
        self.assertEqual((tracking_state.TrackingState.OK, tracking_state.TrackingState.LOST),
                         result.changes[3])
        self.assertEqual((tracking_state.TrackingState.LOST, tracking_state.TrackingState.NOT_INITIALIZED),
                         result.changes[3.3333])
        self.assertEqual((tracking_state.TrackingState.LOST, tracking_state.TrackingState.OK),
                         result.changes[3.6667])

    def test_benchmark_associates_results(self):
        trial_result = MockTrialResult(tracking_states={
            1.3433: tracking_state.TrackingState.NOT_INITIALIZED,
            1.6767: tracking_state.TrackingState.OK,
            1.99: tracking_state.TrackingState.LOST,
            2.3433: tracking_state.TrackingState.NOT_INITIALIZED,
            2.6767: tracking_state.TrackingState.OK,
            3.01: tracking_state.TrackingState.LOST,
            3.3233: tracking_state.TrackingState.NOT_INITIALIZED,
            3.6767: tracking_state.TrackingState.OK,
            4.01: tracking_state.TrackingState.LOST
        })
        reference_trial_result = MockTrialResult(tracking_states={
            1.3333: tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: tracking_state.TrackingState.NOT_INITIALIZED,
            2: tracking_state.TrackingState.NOT_INITIALIZED,
            2.3333: tracking_state.TrackingState.OK,
            2.6667: tracking_state.TrackingState.OK,
            3: tracking_state.TrackingState.OK,
            3.3333: tracking_state.TrackingState.LOST,
            3.6667: tracking_state.TrackingState.LOST,
            4: tracking_state.TrackingState.LOST
        })
        benchmark = track_comp.TrackingComparisonBenchmark()
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertNotIsInstance(result, arvet.core.benchmark.FailedBenchmark)

        self.assertEqual(6, len(result.changes))
        self.assertEqual((tracking_state.TrackingState.NOT_INITIALIZED, tracking_state.TrackingState.OK),
                         result.changes[1.6667])
        self.assertEqual((tracking_state.TrackingState.NOT_INITIALIZED, tracking_state.TrackingState.LOST),
                         result.changes[2])
        self.assertEqual((tracking_state.TrackingState.OK, tracking_state.TrackingState.NOT_INITIALIZED),
                         result.changes[2.3333])
        self.assertEqual((tracking_state.TrackingState.OK, tracking_state.TrackingState.LOST),
                         result.changes[3])
        self.assertEqual((tracking_state.TrackingState.LOST, tracking_state.TrackingState.NOT_INITIALIZED),
                         result.changes[3.3333])
        self.assertEqual((tracking_state.TrackingState.LOST, tracking_state.TrackingState.OK),
                         result.changes[3.6667])

    def test_benchmark_fails_for_not_enough_matching_keys(self):
        trial_result = MockTrialResult(tracking_states={
            1.4333: tracking_state.TrackingState.NOT_INITIALIZED,
            1.7667: tracking_state.TrackingState.OK,
            1.9: tracking_state.TrackingState.LOST,
            2.4333: tracking_state.TrackingState.NOT_INITIALIZED,
            2.7667: tracking_state.TrackingState.OK,
            3.1: tracking_state.TrackingState.LOST,
            3.2333: tracking_state.TrackingState.NOT_INITIALIZED,
            3.7667: tracking_state.TrackingState.OK,
            4.1: tracking_state.TrackingState.LOST
        })
        reference_trial_result = MockTrialResult(tracking_states={
            1.3333: tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: tracking_state.TrackingState.NOT_INITIALIZED,
            2: tracking_state.TrackingState.NOT_INITIALIZED,
            2.3333: tracking_state.TrackingState.OK,
            2.6667: tracking_state.TrackingState.OK,
            3: tracking_state.TrackingState.OK,
            3.3333: tracking_state.TrackingState.LOST,
            3.6667: tracking_state.TrackingState.LOST,
            4: tracking_state.TrackingState.LOST
        })
        benchmark = track_comp.TrackingComparisonBenchmark()
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertIsInstance(result, arvet.core.benchmark.FailedBenchmark)

    def test_offset_adjusts_timestamps(self):
        trial_result = MockTrialResult(tracking_states={
            101.3433: tracking_state.TrackingState.NOT_INITIALIZED,
            101.6767: tracking_state.TrackingState.OK,
            101.99: tracking_state.TrackingState.LOST,
            102.3433: tracking_state.TrackingState.NOT_INITIALIZED,
            102.6767: tracking_state.TrackingState.OK,
            103.01: tracking_state.TrackingState.LOST,
            103.3233: tracking_state.TrackingState.NOT_INITIALIZED,
            103.6767: tracking_state.TrackingState.OK,
            104.01: tracking_state.TrackingState.LOST
        })
        reference_trial_result = MockTrialResult(tracking_states={
            1.3333: tracking_state.TrackingState.NOT_INITIALIZED,
            1.6667: tracking_state.TrackingState.NOT_INITIALIZED,
            2: tracking_state.TrackingState.NOT_INITIALIZED,
            2.3333: tracking_state.TrackingState.OK,
            2.6667: tracking_state.TrackingState.OK,
            3: tracking_state.TrackingState.OK,
            3.3333: tracking_state.TrackingState.LOST,
            3.6667: tracking_state.TrackingState.LOST,
            4: tracking_state.TrackingState.LOST
        })

        # Perform the benchmark, this should fail
        benchmark = track_comp.TrackingComparisonBenchmark()
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertIsInstance(result, arvet.core.benchmark.FailedBenchmark)

        # Adjust the offset, this should work
        benchmark.offset = 100  # Updates the reference timestamps to match the query ones
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertNotIsInstance(result, arvet.core.benchmark.FailedBenchmark)

        self.assertEqual(6, len(result.changes))
        self.assertEqual((tracking_state.TrackingState.NOT_INITIALIZED, tracking_state.TrackingState.OK),
                         result.changes[1.6667])
        self.assertEqual((tracking_state.TrackingState.NOT_INITIALIZED, tracking_state.TrackingState.LOST),
                         result.changes[2])
        self.assertEqual((tracking_state.TrackingState.OK, tracking_state.TrackingState.NOT_INITIALIZED),
                         result.changes[2.3333])
        self.assertEqual((tracking_state.TrackingState.OK, tracking_state.TrackingState.LOST),
                         result.changes[3])
        self.assertEqual((tracking_state.TrackingState.LOST, tracking_state.TrackingState.NOT_INITIALIZED),
                         result.changes[3.3333])
        self.assertEqual((tracking_state.TrackingState.LOST, tracking_state.TrackingState.OK),
                         result.changes[3.6667])

    def test_max_difference_affects_associations(self):
        trial_result = MockTrialResult(tracking_states={
            1.4333: tracking_state.TrackingState.NOT_INITIALIZED,
            11.7667: tracking_state.TrackingState.OK,
            21.9: tracking_state.TrackingState.LOST,
        })
        reference_trial_result = MockTrialResult(tracking_states={
            2.3333: tracking_state.TrackingState.LOST,
            10.6667: tracking_state.TrackingState.LOST,
            20: tracking_state.TrackingState.OK
        })

        # Perform the benchmark, this should fail since the keys are far appart
        benchmark = track_comp.TrackingComparisonBenchmark()
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertIsInstance(result, arvet.core.benchmark.FailedBenchmark)

        # Adjust the max difference, this should now allow associations between
        benchmark.max_difference = 5
        result = benchmark.compare_trial_results(trial_result, reference_trial_result)
        self.assertNotIsInstance(result, arvet.core.benchmark.FailedBenchmark)
