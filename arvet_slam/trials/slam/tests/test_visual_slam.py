# Copyright (c) 2019, John Skinner
import numpy as np
import unittest
from arvet.util.transform import Transform
from arvet.core.sequence_type import ImageSequenceType
import arvet.database.tests.database_connection as dbconn
import arvet.core.tests.mock_types as mock_types
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult, FrameResult


class TestTrialResultDatabase(unittest.TestCase):
    system = None
    image_source = None

    @classmethod
    def setUpClass(cls):
        dbconn.connect_to_test_db()
        cls.system = mock_types.MockSystem()
        cls.image_source = mock_types.MockImageSource()
        cls.system.save()
        cls.image_source.save()

    def setUp(self):
        # Remove the collection as the start of the test, so that we're sure it's empty
        SLAMTrialResult._mongometa.collection.drop()

    @classmethod
    def tearDownClass(cls):
        # Clean up after ourselves by dropping the collection for this model
        SLAMTrialResult._mongometa.collection.drop()
        mock_types.MockSystem._mongometa.collection.drop()
        mock_types.MockImageSource._mongometa.collection.drop()

    def test_stores_and_loads(self):
        results = [
            FrameResult(
                timestamp=idx + np.random.normal(0, 0.01),
                pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0)
                ),
                motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0)
                ),
                estimated_pose=Transform(
                    (idx * 15 + np.random.normal(0, 1), idx + np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0)
                ),
                estimated_motion=Transform(
                    (np.random.normal(0, 1), np.random.normal(0, 0.1), np.random.normal(0, 1)),
                    (1, 0, 0, 0)
                ),
                tracking_state=TrackingState.OK,
                num_features=np.random.randint(10, 1000),
                num_matches=np.random.randint(10, 1000)
            )
            for idx in range(10)
        ]
        obj = SLAMTrialResult(
            system=self.system,
            image_source=self.image_source,
            success=True,
            settings={'key': 'value'},
            results=results,
            has_scale=True,
            sequence_type=ImageSequenceType.INTERACTIVE
        )
        obj.save()

        # Load all the entities
        all_entities = list(SLAMTrialResult.objects.all())
        self.assertGreaterEqual(len(all_entities), 1)
        self.assertEqual(all_entities[0], obj)
        all_entities[0].delete()
