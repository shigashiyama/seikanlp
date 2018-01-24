import constants


def is_segmentation_task(task):
    if (task == constants.TASK_SEG or
        task == constants.TASK_SEGTAG or
        task == constants.TASK_DUAL_SEG or
        task == constants.TASK_DUAL_SEGTAG):
        return True
    else:
        return False


def is_tagging_task(task):
    if (task == constants.TASK_TAG or
        task == constants.TASK_SEGTAG or
        task == constants.TASK_DUAL_TAG or
        task == constants.TASK_DUAL_SEGTAG):
        return True
    else:
        return False


def is_parsing_task(task):
    return task == constants.TASK_DEP or task == constants.TASK_TDEP


def is_typed_parsing_task(task):
    return task == constants.TASK_TDEP


def is_attribute_annotation_task(task):
    return task == constants.TASK_ATTR


def is_single_st_task(task):
    if (task == constants.TASK_SEG or
        task == constants.TASK_SEGTAG or
        task == constants.TASK_TAG):
        return True
    else:
        return False


def is_dual_st_task(task):
    if (task == constants.TASK_DUAL_SEG or
        task == constants.TASK_DUAL_SEGTAG or
        task == constants.TASK_DUAL_TAG):
        return True
    else:
        return False
