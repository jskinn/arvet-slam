import logging
import time
from itertools import chain, zip_longest
from pymongo import UpdateMany
from pymodm.context_managers import no_auto_dereference
from arvet.core.system import VisionSystem
from arvet.core.trial_result import TrialResult
from arvet.core.image import Image
from arvet.core.image_source import ImageSource
from arvet.core.metric import Metric
from arvet.database.autoload_modules import autoload_modules
from arvet_slam.metrics.frame_error.frame_error_result import FrameError, FrameErrorResult, json_value


def update_frame_errors_system_properties():
    # Start with the frame error result, which will tell us which trial results have been measured
    start_time = time.time()
    logging.getLogger(__name__).info("Updating 'system_properties' for all FrameError objects ...")
    logging.getLogger(__name__).info("Getting set of trial result ids...")
    trial_result_ids = FrameErrorResult._mongometa.collection.distinct('trial_results')

    # Get the unique system ids
    logging.getLogger(__name__).info(f"Getting set of system ids ({time.time() - start_time})...")
    system_ids = TrialResult._mongometa.collection.distinct('system')

    # Autoload the types for the trial and systems
    logging.getLogger(__name__).info(f"Autoloading types ({time.time() - start_time})...")
    autoload_modules(VisionSystem, ids=system_ids)
    autoload_modules(TrialResult, ids=trial_result_ids)

    # Get the set of system ids used in those trials
    logging.getLogger(__name__).info(f"Updating information from {len(trial_result_ids)} trials "
                                     f"over {len(system_ids)} systems...")
    n_updated = 0
    for system_id in system_ids:
        system = VisionSystem.objects.get({'_id': system_id})
        for trial_result in TrialResult.objects.raw({
            '_id': {'$in': trial_result_ids}, 'system': system_id
        }).only('_id', 'settings'):
            system_properties = system.get_properties(None, trial_result.settings)
            n_updated += FrameError.objects.raw({'trial_result': trial_result.pk}).update({
                '$set': {'system_properties': {str(k): json_value(v) for k, v in system_properties.items()}}
            }, upsert=False)  # We definitely don't want to upsert
    logging.getLogger(__name__).info(f"Updated {n_updated} FrameError objects in {time.time() - start_time}s ")


def update_frame_error_image_information(only_missing: bool = False, batch_size: int = 5000):
    """
    Update
    :return:
    """
    # Find the ids
    start_time = time.time()
    logging.getLogger(__name__).info("Updating 'image_properties' for all FrameError objects ...")
    logging.getLogger(__name__).info("Loading set of referenced image IDs...")
    if only_missing:
        frame_errors = FrameError.objects.raw({
            'image_properties': {'$exists': False}
        }).only('image').values()
        num_images = frame_errors.count()
    else:
        frame_errors = FrameError.objects.all().only('image').values()
        num_images = frame_errors.count()

    # Work out how many batches to do. Images will be loaded in a batch to create a single bulk_write
    logging.getLogger(__name__).info(f"Found {num_images} errors, "
                                     f"updating information {batch_size} frame results at a time "
                                     f"({time.time() - start_time}s) ...")
    n_updated = 0
    updates_sent = 0
    completed_ids = set()
    for frame_error_objects in grouper(frame_errors, batch_size):
        # For each image, update all frame error objects that link to it
        batch_ids = [error_obj['image'] for error_obj in frame_error_objects if error_obj['image'] not in completed_ids]
        if len(batch_ids) > 0:
            completed_ids = completed_ids | set(batch_ids)
            images = Image.objects.raw({'_id': {'$in': batch_ids}})
            write_operations = [
                UpdateMany(
                    {'image': image.pk},
                    {'$set': {'image_properties': {str(k): json_value(v) for k, v in image.get_properties().items()}}}
                )
                for image in images
            ]
            result = FrameError._mongometa.collection.bulk_write(write_operations, ordered=False)
            n_updated += result.modified_count
            updates_sent += len(write_operations)
            logging.getLogger(__name__).info(
                f"Updates sent for {updates_sent} images, updating {n_updated} FrameErrors"
                f" in {time.time() - start_time}s ...")
    logging.getLogger(__name__).info(f"Updated {n_updated} FrameError objects in {time.time() - start_time}s ")


def update_frame_error_result_image_source_properties():
    # Start by finding the image source ids
    start_time = time.time()
    logging.getLogger(__name__).info("Updating 'image_source_properties' for all FrameErrorResult objects ...")
    logging.getLogger(__name__).info("Loading set of referenced image source IDs...")
    image_source_ids = FrameErrorResult._mongometa.collection.distinct('image_source')

    # Autoload the image source type
    logging.getLogger(__name__).info(f"Autoloading image source types ({time.time() - start_time}s) ...")
    autoload_modules(ImageSource, ids=image_source_ids)

    logging.getLogger(__name__).info(f"Found {len(image_source_ids)} image sources, updating information "
                                     f"({time.time() - start_time}s) ...")
    n_updated = 0
    for image_source_id in image_source_ids:
        image_source = ImageSource.objects.get({'_id': image_source_id})
        image_source_properties = image_source.get_properties()
        n_updated += FrameErrorResult.objects.raw({'image_source': image_source_id}).update({
            '$set': {'image_source_properties': {str(k): json_value(v) for k, v in image_source_properties.items()}}
        }, upsert=False)
    logging.getLogger(__name__).info(f"Updated {n_updated} FrameErrorResult objects in {time.time() - start_time}s ")


def update_frame_error_result_metric_properties():
    # Start by finding the image source ids
    start_time = time.time()
    logging.getLogger(__name__).info("Updating 'metric_properties' for all FrameErrorResult objects ...")
    logging.getLogger(__name__).info("Loading set of referenced metric IDs...")
    metric_ids = FrameErrorResult._mongometa.collection.distinct('metric')

    # Autoload the image source type
    logging.getLogger(__name__).info(f"Autoloading metric types ({time.time() - start_time}s) ...")
    autoload_modules(Metric, ids=metric_ids)

    logging.getLogger(__name__).info(f"Found {len(metric_ids)} metrics, updating information "
                                     f"({time.time() - start_time}s) ...")
    n_updated = 0
    for metric_id in metric_ids:
        metric = Metric.objects.get({'_id': metric_id})
        metric_properties = metric.get_properties()
        n_updated += FrameErrorResult.objects.raw({'metric': metric_id}).update({
            '$set': {'metric_properties': {str(k): json_value(v) for k, v in metric_properties.items()}}
        }, upsert=False)
    logging.getLogger(__name__).info(f"Updated {n_updated} FrameErrorResult objects in {time.time() - start_time}s ")


def update_frame_error_result_columns():
    start_time = time.time()
    logging.getLogger(__name__).info("Updating 'frame_columns' for all FrameErrorResult objects ...")
    n_updated = 0
    for metric_result in FrameErrorResult.objects.all().only('_id', 'errors'):
        # Collect all the frame error ids referenced by this metric result
        frame_error_ids = []
        for trial_errors in metric_result.errors:
            with no_auto_dereference(type(trial_errors)):
                frame_error_ids.extend(trial_errors.frame_errors)

        # Load only a partial object for each frame error
        frame_errors = FrameError.objects.raw({
            '_id': {'$in': frame_error_ids}
        }).only('system_properties', 'image_properties')

        # Produce the combined list of available columns from the frame errors
        frame_error_columns = list(set(chain.from_iterable(
            frame_error.get_columns()
            for frame_error in frame_errors
        )))
        # Send only the updated frame columns to the server
        n_updated += FrameErrorResult.objects.raw({'_id': metric_result.pk}).update({
            '$set': {'frame_columns': frame_error_columns}
        }, upsert=False)
    logging.getLogger(__name__).info(f"Updated {n_updated} FrameErrorResult objects in {time.time() - start_time}s ")


def grouper(iterable, n):
    """
    Collect data into fixed-length chunks or blocks
    Adjusted from the itertools recipes in the documentation
    :param iterable:
    :param n:
    :return:
    """
    args = [iter(iterable)] * n
    return (
        [e for e in group if e is not None]
        for group in zip_longest(*args, fillvalue=None)
    )
