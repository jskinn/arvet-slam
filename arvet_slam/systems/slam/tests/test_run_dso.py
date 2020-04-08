# Copyright (c) 2017, John Skinner
import unittest
import numpy as np
from pymodm.context_managers import no_auto_dereference

from arvet.util.test_helpers import ExtendedTestCase
from arvet.core.sequence_type import ImageSequenceType
from arvet.metadata.camera_intrinsics import CameraIntrinsics
from arvet_slam.trials.slam.tracking_state import TrackingState
from arvet_slam.trials.slam.visual_slam import SLAMTrialResult
from arvet_slam.systems.slam.direct_sparse_odometry import DSO, RectificationMode
from arvet_slam.systems.test_helpers.demo_image_builder import DemoImageBuilder, ImageMode


original_trans_variation = [
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.2097630837054241e-07,
    1.333140038924898e-07, 1.7676393324148007e-07, 0.000667368920874547, 0.0006835777207499621, 0.016545248721387126,
    0.012078357177169107, 0.009303794785921457, 0.011355947424553394, 0.010492094524467668, 0.007938817520961013,
    0.0366866086825253, 0.00669786479865889, 0.004720993373575399, 0.012849503543237408, 0.03824718717573727,
    0.00905208947384373, 0.017151877199130333, 0.015109745437679125, 0.027607008293560952, 0.030224901105104396,
    0.07728446730675756, 0.015803492392404722, 0.016396613011190296, 0.03832649807710987, 0.039233412229279084,
    0.04331342024828073, 0.04796933242469964, 0.02572029299432036, 0.01150915975757924, 0.015661450403252727,
    0.010703770281345209, 0.060837212058723375, 0.06043669635753965, 0.041396021760863363, 0.03788099126535733,
    0.012247933854418726, 0.031077112060348005, 0.05750558698007892, 0.02261904024581898, 0.06583547505083352,
    0.028763432306732624, 0.011709712318305952, 0.03726296117117693, 0.02482250058950522, 0.062463348656246524,
    0.04241186627540177, 0.016331589034384093, 0.05331883594960236, 0.013847353525823506, 0.03132084005320759,
    0.0165998337006255, 0.038316728936377165, 0.01383674426830992, 0.020121288909278716, 0.04000747875174828,
    0.031123916600733128, 0.020953729319316414, 0.024315779521491835, 0.03373999757748907, 0.022240626311640695,
    0.02260481667575948, 0.03338236654696881, 0.027653128908481532, 0.02735397664179108, 0.04037575275037536,
    0.03658945043641235, 0.020554760203670256, 0.029920477623719684, 0.03830195563383777, 0.01969924765490245,
    0.028431582323131074, 0.027019829465404623, 0.02088332705534978, 0.025059977281558062, 0.03792319716814306,
    0.007016016688992685, 0.026813293633884797, 0.008917489503455134, 0.0236440900284801, 0.01994071345414773,
    0.020134790718555524, 0.050535474560315506, 0.010039191919981637, 0.017209065288162417, 0.0023213680483357806,
    0.029670128651846334, 0.02185464358192474, 0.010224492335844498
]
original_rot_variation = [
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 1.4749405661148783e-08,
    1.2032164243271195e-08, 1.061450583022483e-08, 6.127883008981156e-05, 5.971278510900896e-05, 0.000268859930437184,
    0.0002896823303896984, 0.0002852114022412718, 0.00038005806625589347, 0.00029005711149412077, 0.0004269629392716357,
    0.0014833618300077994, 0.0007538993164638089, 0.00013662222103745422, 0.0002471543969644325, 0.0015385046780403348,
    3.729729904160701e-05, 0.00011964147478726909, 0.000751700590768373, 0.0006601521409013676, 0.0008663431587163468,
    0.0006379635584917878, 0.0003753230563154773, 0.0007863084577328723, 0.0003639894033492235, 0.0008832204357680101,
    0.000686318838431783, 0.0016191096737836067, 0.0005718175930106968, 0.0006162415365616931, 0.0004255728614955088,
    0.0006776666232813801, 0.00043878566006371427, 0.0005105962771300014, 0.0012061351235620264, 0.001053345057192446,
    0.0006251527773486635, 0.00033203039037854576, 0.0008293396744208073, 0.00040276664029996044,
    0.00044375831533867394, 0.0006818188860610928, 0.0004610670721521412, 0.0003533135687248293, 0.0006419527618281076,
    0.0008321669183449238, 0.0004855371628984432, 0.0011042838311380579, 0.0010180813226586993, 0.0006767189010667486,
    0.00044913939678430754, 0.000868181412912226, 0.0010343264992129968, 0.0007900826797434999, 0.001371769646297577,
    0.0005195610311906099, 0.0008584023143606269, 0.0011350364031343036, 0.0002597051641284233, 0.0005753556921284848,
    0.0007696223455737479, 0.0006654301859478378, 0.001168166383310357, 0.0004722605695768369, 0.0012203201071618343,
    0.0006733017390598073, 0.0014993830824999782, 0.00041665031444393646, 0.0008988567920662894, 0.0023786957805294025,
    0.001306733422391217, 0.0006503602697141871, 0.0013046007937852295, 6.653293111896898e-05, 0.00019914010877704208,
    0.0005101790551902905, 0.0005138486315729264, 0.000730172611812625, 0.0007513185250569957, 0.000342435338733216,
    0.0006994401157951645, 0.0010808256515359672, 0.0018261240167251668, 0.0006830576329555502, 0.001260564216596473,
    0.0005906456966589635, 0.0008545572086839467, 0.0006083681239312961, 0.0010121523752786008
]
original_timing_1 = [
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.007042407989501953,
    0.05229663848876953, 0.004961729049682617, 0.005233049392700195, 0.005128622055053711, 0.004435539245605469,
    0.0050427913665771484, 0.005001544952392578, 0.00608515739440918, 0.005109310150146484, 0.004861116409301758,
    0.0047338008880615234, 0.005100727081298828, 0.005972146987915039, 0.005826234817504883, 0.06856131553649902,
    0.00679469108581543, 0.0056705474853515625, 0.00511479377746582, 0.005209445953369141, 0.005374431610107422,
    0.005784511566162109, 0.005205392837524414, 0.004915714263916016, 0.005587577819824219, 0.08084845542907715,
    0.005090475082397461, 0.006021976470947266, 0.005448579788208008, 0.006515979766845703, 0.007836103439331055,
    0.006039857864379883, 0.006329536437988281, 0.005768537521362305, 0.006494045257568359, 0.006479024887084961,
    0.0057675838470458984, 0.0066776275634765625, 0.007512331008911133, 0.007184028625488281, 0.006215095520019531,
    0.005462646484375, 0.005570173263549805, 0.005533456802368164, 0.0058743953704833984, 0.006162405014038086,
    0.0053255558013916016, 0.0055882930755615234, 0.008016824722290039, 0.006810903549194336, 0.005294322967529297,
    0.0057830810546875, 0.005508899688720703, 0.006101846694946289, 0.0063250064849853516, 0.006762981414794922,
    0.006104707717895508, 0.0069141387939453125, 0.006140232086181641, 0.005381584167480469, 0.007395744323730469,
    0.008162260055541992, 0.005930423736572266, 0.005994319915771484, 0.005839109420776367, 0.005472660064697266,
    0.005158424377441406, 0.00672459602355957, 0.005649566650390625, 0.006064891815185547, 0.0060460567474365234,
    0.0060803890228271484, 0.006324052810668945, 0.005621194839477539, 0.006171703338623047, 0.006699562072753906,
    0.007557868957519531, 0.006952762603759766, 0.006691932678222656, 0.006562948226928711, 0.006021022796630859,
    0.006058454513549805, 0.006620883941650391, 0.007990598678588867, 0.005879878997802734, 0.006666898727416992,
    0.006415605545043945, 0.005921840667724609, 0.0065424442291259766, 0.005631685256958008
]
original_timing_2 = [
    np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, 0.007272481918334961,
    0.05047321319580078, 0.005101442337036133, 0.005648374557495117, 0.004644155502319336, 0.004477024078369141,
    0.004744291305541992, 0.004559516906738281, 0.004926443099975586, 0.004858732223510742, 0.004479646682739258,
    0.00469207763671875, 0.004671812057495117, 0.0056264400482177734, 0.006244659423828125, 0.06737661361694336,
    0.00682830810546875, 0.005547761917114258, 0.0051538944244384766, 0.0053598880767822266, 0.00519251823425293,
    0.005692481994628906, 0.00526881217956543, 0.004933357238769531, 0.006033420562744141, 0.08112406730651855,
    0.005381107330322266, 0.00555109977722168, 0.005705356597900391, 0.006777048110961914, 0.007879495620727539,
    0.005842924118041992, 0.006042957305908203, 0.005427360534667969, 0.006373882293701172, 0.0061800479888916016,
    0.006003141403198242, 0.005995750427246094, 0.0074100494384765625, 0.00678253173828125, 0.006491899490356445,
    0.0054340362548828125, 0.005736351013183594, 0.005402326583862305, 0.005833625793457031, 0.0057713985443115234,
    0.005239009857177734, 0.005204200744628906, 0.005553245544433594, 0.00638580322265625, 0.005471706390380859,
    0.006062507629394531, 0.0058901309967041016, 0.0065381526947021484, 0.006651163101196289, 0.0070612430572509766,
    0.005815267562866211, 0.006832122802734375, 0.0061070919036865234, 0.005293130874633789, 0.006663799285888672,
    0.0077130794525146484, 0.0057201385498046875, 0.005517244338989258, 0.005543708801269531, 0.005791187286376953,
    0.005365848541259766, 0.0066661834716796875, 0.006243467330932617, 0.005821704864501953, 0.00638127326965332,
    0.005600929260253906, 0.0058743953704833984, 0.005454063415527344, 0.006028175354003906, 0.0066525936126708984,
    0.0076444149017333984, 0.006854534149169922, 0.006414651870727539, 0.0067768096923828125, 0.006209850311279297,
    0.006081819534301758, 0.006468772888183594, 0.007743358612060547, 0.005644559860229492, 0.006044149398803711,
    0.0060977935791015625, 0.00576019287109375, 0.006875276565551758, 0.005555868148803711
]


class TestRunDSO(ExtendedTestCase):

    def test_simple_trial_run_rect_none(self):
        # Actually run the system using mocked images
        num_frames = 100
        max_time = 50
        speed = 0.1

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            seed=0,
            width=640, height=480, num_stars=100,
            length=max_time * speed, speed=speed,
            close_ratio=0.5, min_size=10, max_size=200
        )

        subject = DSO(
            rectification_mode=RectificationMode.NONE,
            # These should be irrelevant
            rectification_intrinsics=CameraIntrinsics(
                width=320,
                height=240,
                fx=160,
                fy=160,
                cx=160,
                cy=120
            )
        )
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result = subject.finish_trial()

        self.assertIsInstance(result, SLAMTrialResult)
        with no_auto_dereference(SLAMTrialResult):
            self.assertEqual(subject.pk, result.system)
        self.assertTrue(result.success)
        self.assertFalse(result.has_scale)
        self.assertIsNotNone(result.run_time)
        self.assertIsNotNone(result.settings)
        self.assertEqual(num_frames, len(result.results))

        has_been_found = False
        for idx, frame_result in enumerate(result.results):
            self.assertEqual(max_time * idx / num_frames, frame_result.timestamp)
            self.assertIsNotNone(frame_result.pose)
            self.assertIsNotNone(frame_result.motion)
            if frame_result.tracking_state is TrackingState.OK:
                has_been_found = True
        self.assertTrue(has_been_found)

    def test_simple_trial_run_rect_calib(self):
        # Actually run the system using mocked images
        num_frames = 100
        max_time = 50
        speed = 0.1

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            seed=0,
            width=640, height=480, num_stars=100,
            length=max_time * speed, speed=speed,
            close_ratio=0.5, min_size=10, max_size=200
        )

        subject = DSO(
            rectification_mode=RectificationMode.CALIB,
            rectification_intrinsics=CameraIntrinsics(
                width=320,
                height=240,
                fx=160,
                fy=160,
                cx=160,
                cy=120
            )
        )
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result = subject.finish_trial()

        self.assertIsInstance(result, SLAMTrialResult)
        with no_auto_dereference(SLAMTrialResult):
            self.assertEqual(subject.pk, result.system)
        self.assertTrue(result.success)
        self.assertFalse(result.has_scale)
        self.assertIsNotNone(result.run_time)
        self.assertIsNotNone(result.settings)
        self.assertEqual(num_frames, len(result.results))

        has_been_found = False
        for idx, frame_result in enumerate(result.results):
            self.assertEqual(max_time * idx / num_frames, frame_result.timestamp)
            self.assertIsNotNone(frame_result.pose)
            self.assertIsNotNone(frame_result.motion)
            if frame_result.tracking_state is TrackingState.OK:
                has_been_found = True
        self.assertTrue(has_been_found)

    def test_simple_trial_run_rect_crop(self):
        # Actually run the system using mocked images
        num_frames = 100
        max_time = 50
        speed = 0.1

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            seed=0,
            width=640, height=480, num_stars=100,
            length=max_time * speed, speed=speed,
            close_ratio=0.5, min_size=10, max_size=200
        )

        # image_builder.visualise_sequence(max_time, max_time / num_frames)
        # return

        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=CameraIntrinsics(
                width=480,
                height=480,
                fx=240,
                fy=240,
                cx=240,
                cy=240
            )
        )
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result = subject.finish_trial()

        self.assertIsInstance(result, SLAMTrialResult)
        with no_auto_dereference(SLAMTrialResult):
            self.assertEqual(subject.pk, result.system)
        self.assertTrue(result.success)
        self.assertFalse(result.has_scale)
        self.assertIsNotNone(result.run_time)
        self.assertIsNotNone(result.settings)
        self.assertEqual(num_frames, len(result.results))

        has_been_found = False
        for idx, frame_result in enumerate(result.results):
            self.assertEqual(max_time * idx / num_frames, frame_result.timestamp)
            self.assertIsNotNone(frame_result.pose)
            self.assertIsNotNone(frame_result.motion)
            if frame_result.tracking_state is TrackingState.OK:
                has_been_found = True
        self.assertTrue(has_been_found)

    @unittest.skip("Tends to segfault if running as part of a suite.")
    def test_simple_trial_run_rect_crop_larger_than_source(self):
        # Actually run the system using mocked images
        num_frames = 100
        max_time = 50
        speed = 0.1

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            seed=0,
            width=1242, height=376,     # These are the dimensions of the KITTI dataset
            num_stars=100,
            length=max_time * speed, speed=speed,
            close_ratio=0.5, min_size=10, max_size=200
        )

        subject = DSO(
            rectification_mode=RectificationMode.CROP,
            rectification_intrinsics=CameraIntrinsics(
                width=480,
                height=480,     # This is larger than the input height
                fx=240,
                fy=240,
                cx=240,
                cy=240
            )
        )
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result = subject.finish_trial()

        self.assertIsInstance(result, SLAMTrialResult)
        with no_auto_dereference(SLAMTrialResult):
            self.assertEqual(subject.pk, result.system)
        self.assertTrue(result.success)
        self.assertFalse(result.has_scale)
        self.assertIsNotNone(result.run_time)
        self.assertIsNotNone(result.settings)
        self.assertEqual(num_frames, len(result.results))

        has_been_found = False
        self.assertEqual(num_frames, len(result.results))
        for idx, frame_result in enumerate(result.results):
            self.assertEqual(max_time * idx / num_frames, frame_result.timestamp)
            self.assertIsNotNone(frame_result.pose)
            self.assertIsNotNone(frame_result.motion)
            if frame_result.tracking_state is TrackingState.OK:
                has_been_found = True
        self.assertTrue(has_been_found)

    def test_run_long(self):
        # Actually run the system using mocked images
        num_frames = 1000
        max_time = 50
        speed = 0.1

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            seed=0,
            width=640, height=480, num_stars=100,
            length=max_time * speed, speed=speed,
            close_ratio=0.5, min_size=10, max_size=200
        )

        subject = DSO(
            rectification_mode=RectificationMode.NONE,
            rectification_intrinsics=image_builder.get_camera_intrinsics()
        )
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result = subject.finish_trial()

        has_been_found = False
        self.assertEqual(num_frames, len(result.results))
        for idx, frame_result in enumerate(result.results):
            self.assertEqual(max_time * idx / num_frames, frame_result.timestamp)
            self.assertIsNotNone(frame_result.pose)
            self.assertIsNotNone(frame_result.motion)
            if frame_result.tracking_state is TrackingState.OK:
                has_been_found = True
        self.assertTrue(has_been_found)

    def test_consistency(self):
        # Actually run the system using mocked images
        num_frames = 100
        max_time = 50
        speed = 0.1

        image_builder = DemoImageBuilder(
            mode=ImageMode.MONOCULAR,
            seed=0,
            width=640, height=480, num_stars=100,
            length=max_time * speed, speed=speed,
            close_ratio=0.5, min_size=10, max_size=200
        )

        subject = DSO(
            rectification_mode=RectificationMode.NONE,
            rectification_intrinsics=image_builder.get_camera_intrinsics()
        )
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result1 = subject.finish_trial()

        subject = DSO(
            rectification_mode=RectificationMode.NONE,
            rectification_intrinsics=image_builder.get_camera_intrinsics()
        )
        subject.set_camera_intrinsics(image_builder.get_camera_intrinsics(), max_time / num_frames)

        subject.start_trial(ImageSequenceType.SEQUENTIAL)
        for idx in range(num_frames):
            time = max_time * idx / num_frames
            image = image_builder.create_frame(time)
            subject.process_image(image, time)
        result2 = subject.finish_trial()

        trans_variation = []
        rot_variation = []
        has_any_estimate = False
        self.assertEqual(len(result1.results), len(result2.results))
        for frame_result_1, frame_result_2 in zip(result1.results, result2.results):
            self.assertEqual(frame_result_1.timestamp, frame_result_2.timestamp)
            self.assertEqual(frame_result_1.tracking_state, frame_result_2.tracking_state)
            if frame_result_1.estimated_motion is None or frame_result_2.estimated_motion is None:
                self.assertEqual(frame_result_1.estimated_motion, frame_result_2.estimated_motion)
                trans_variation.append(np.nan)
                rot_variation.append(np.nan)
            else:
                has_any_estimate = True
                motion1 = frame_result_1.estimated_motion
                motion2 = frame_result_2.estimated_motion

                loc_diff = motion1.location - motion2.location
                trans_variation.append(np.linalg.norm(loc_diff))
                self.assertNPClose(loc_diff, np.zeros(3), rtol=0, atol=1e-14)
                quat_diff = motion1.rotation_quat(True) - motion2.rotation_quat(True)
                rot_variation.append(np.linalg.norm(quat_diff))
                self.assertNPClose(quat_diff, np.zeros(4), rtol=0, atol=1e-14)
        self.assertTrue(has_any_estimate)

        # import matplotlib.pyplot as plt
        # print([repr(var) for var in trans_variation])
        # print([repr(var) for var in rot_variation])
        # print([
        #     repr(frame_result.processing_time)
        #     for frame_result in result1.results
        # ])
        # print([
        #     repr(frame_result.processing_time)
        #     for frame_result in result2.results
        # ])
        #
        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # ax1.plot(list(range(len(original_trans_variation))), original_trans_variation)
        # ax1.plot(list(range(len(trans_variation))), trans_variation)
        # ax2.plot(list(range(len(original_rot_variation))), original_rot_variation)
        # ax2.plot(list(range(len(rot_variation))), rot_variation)
        #
        # ax3.plot(list(range(len(original_timing_1))), original_timing_1)
        # ax3.plot(list(range(len(original_timing_2))), original_timing_2)
        # ax3.plot(list(range(len(result1.results))), [
        #     frame_result.processing_time
        #     for frame_result in result1.results
        # ])
        # ax3.plot(list(range(len(result2.results))), [
        #     frame_result.processing_time
        #     for frame_result in result2.results
        # ])
        #
        # plt.show()
