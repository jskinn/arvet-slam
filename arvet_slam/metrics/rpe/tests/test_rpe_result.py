# Copyright (c) 2017, John Skinner
import numpy as np
import unittest
import bson
import pickle
import arvet.util.dict_utils as du
import arvet.database.tests.test_entity as entity_test
import arvet_slam.metrics.rpe.rpe_result as rpe_res


class TestBenchmarkRPEResult(entity_test.EntityContract, unittest.TestCase):

    def get_class(self):
        return rpe_res.BenchmarkRPEResult

    def make_instance(self, *args, **kwargs):
        timestamps = sorted(idx + np.random.normal(0, 1) for idx in range(100))
        kwargs = du.defaults(kwargs, {
            'benchmark_id': np.random.randint(0, 10),
            'trial_result_ids': [bson.ObjectId() for _ in range(4)],
            'timestamps': timestamps,
            'errors': [
                [timestamps[idx - 1], timestamps[idx],
                 np.random.exponential(3), np.random.beta(0.5, 0.75),
                 np.random.exponential(3), np.random.beta(0.5, 0.75)]
                for idx in range(1, len(timestamps))
            ],
            'rpe_settings': {}
        })
        return rpe_res.BenchmarkRPEResult(*args, **kwargs)

    def assert_models_equal(self, benchmark_result1, benchmark_result2):
        """
        Helper to assert that two benchmark results are equal
        :param benchmark_result1: BenchmarkRPEResult
        :param benchmark_result2: BenchmarkRPEResult
        :return:
        """
        if (not isinstance(benchmark_result1, rpe_res.BenchmarkRPEResult) or
                not isinstance(benchmark_result2, rpe_res.BenchmarkRPEResult)):
            self.fail('object was not a BenchmarkRPEResult')
        self.assertEqual(benchmark_result1.identifier, benchmark_result2.identifier)
        self.assertEqual(benchmark_result1.success, benchmark_result2.success)
        self.assertEqual(benchmark_result1.benchmark, benchmark_result2.benchmark)
        self.assertEqual(benchmark_result1.trial_results, benchmark_result2.trial_results)
        self.assertEqual(benchmark_result1.timestamps, benchmark_result2.timestamps)
        self.assertNPEqual(benchmark_result1._errors_observations, benchmark_result2._errors_observations)
        self.assertEqual(benchmark_result1.settings, benchmark_result2.settings)

    def assert_serialized_equal(self, s_model1, s_model2):
        """
        Override assert for serialized models equal to measure the bson more directly.
        :param s_model1: 
        :param s_model2: 
        :return: 
        """
        self.assertEqual(set(s_model1.keys()), set(s_model2.keys()))
        for key in s_model1.keys():
            if key is not 'errors' and key is not 'trial_results':
                self.assertEqual(s_model1[key], s_model2[key])

        # Special case for sets
        self.assertEqual(set(s_model1['trial_results']), set(s_model2['trial_results']))

        # Special case for BSON
        errors1 = pickle.loads(s_model1['errors'])
        errors2 = pickle.loads(s_model2['errors'])
        self.assertEqual(errors1, errors2)

    def assertNPEqual(self, arr1, arr2):
        self.assertTrue(np.array_equal(arr1, arr2), "Arrays {0} and {1} are not equal".format(str(arr1), str(arr2)))

    def assertNPClose(self, arr1, arr2):
        self.assertTrue(np.all(np.isclose(arr1, arr2)), "Arrays {0} and {1} are not close".format(str(arr1), str(arr2)))
