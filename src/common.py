import constants


def is_segmentation_task(task):
    return is_single_segmentation_task(task) #or is_dual_segmentation_task(task)


def is_single_segmentation_task(task):
    if (task == constants.TASK_SEG or task == constants.TASK_SEGTAG):
        return True
    else:
        return False


def is_tagging_task(task):
    if (task == constants.TASK_TAG or task == constants.TASK_SEGTAG):
        return True
    else:
        return False


def is_parsing_task(task):
    return task == constants.TASK_DEP or task == constants.TASK_TDEP


def is_typed_parsing_task(task):
    return task == constants.TASK_TDEP


def is_sequence_labeling_task(task):
    return is_segmentation_task(task) or is_tagging_task(task) #or is_dual_segmentation_task(task)


def get_attribute_values(attr_vals_str, num=0):
    if attr_vals_str:
        ret = [int(val) for val in attr_vals_str.split(constants.ATTR_INFO_DELIM)]
        for i in range(num - len(ret)):
            ret.append(0)
    else:
        ret = [0 for i in range(num)]
    return ret


def get_attribute_boolvalues(attr_vals_str, num=0):
    if attr_vals_str:
        ret = [val.lower() == 't' for val in attr_vals_str.split(constants.ATTR_INFO_DELIM)]
        for i in range(num - len(ret)):
            ret.append(False)
    else:
        ret = [False for i in range(num)]
    return ret


def get_attribute_labelsets(arg, num=0):
    if arg:
        tmp = [vals for vals in arg.split(constants.ATTR_INFO_DELIM3)]
        ret = [set(vals.split(constants.ATTR_INFO_DELIM2)) for vals in tmp]
        for i in range(num - len(ret)):
            ret.append(set())
        return ret
    else:
        return [set() for i in range(num)]


def use_fmeasure(keys):
    for key in keys:
        if key.startswith('B-'):
            return True

    return False
