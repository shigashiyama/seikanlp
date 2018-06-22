import constants


def is_segmentation_task(task):
    if (task == constants.TASK_SEG or
        task == constants.TASK_SEGTAG or
        task == constants.TASK_HSEG):
        return True
    else:
        return False


def is_tagging_task(task):
    if (task == constants.TASK_TAG or
        task == constants.TASK_SEGTAG):
        return True
    else:
        return False


def is_parsing_task(task):
    return task == constants.TASK_DEP or task == constants.TASK_TDEP


def is_typed_parsing_task(task):
    return task == constants.TASK_TDEP


def is_single_st_task(task):
    return is_segmentation_task(task) or is_tagging_task(task)


def get_attribute_values(attr_vals_str):
    if attr_vals_str:
        return [int(val) for val in attr_vals_str.split(constants.ATTR_INFO_DELIM2)]
    else:
        return []


def get_attribute_boolvalues(attr_vals_str):
    if attr_vals_str:
        ret = [val.lower() == 't' for val in attr_vals_str.split(constants.ATTR_INFO_DELIM2)]
        if ret[0] is False:
            ret[0] = True
        return ret    # regard the first attribute as a target of prediction
    else:
        return [True] # regard the first attribute as a target of prediction


def get_attribute_labelsets(arg, num):
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
