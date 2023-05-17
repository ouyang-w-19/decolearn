"""
This script contains methods that format string in various format into the target format
Contains all of the helper methods of LSBoost
"""
import numpy as np
import math as m
from typing import Union, List


def replace_sub_string(string, sub_start, sub_end, payload_string) -> str:
    """
    - Replace a spec. reoccurring sub-string with pos. in [sub_start,sub_end]
    :param string: Original string
    :param sub_start: Defines the start position of the sub to be replaced
    :param sub_end: Defines the end position of the sub to be replaced
    :param payload_string: The sub-string that replaces the target
    :return: New string with target replaced by the given payload
    """
    start_idx = string.find(sub_start)
    end_idx = string.find(sub_end, start_idx) + 1
    target_string = string[start_idx:end_idx]

    new_string = string.replace(target_string, payload_string)

    return new_string


def hierarch_dict(d: dict, indent=0):
    """
    - Function that prints a nested dict in a hierarchical
    :param d: Nested d to be printed
    :param indent:  Indent at start (level 0)
    """
    for key, value in d.items():
        print('\t' * indent + str(key))
        if isinstance(value, dict):
            hierarch_dict(value, indent+1)
        else:
            print('\t' * (indent+1) + str(value))


def sub_to_sub(string: str, sub_start: str, sub_end: str, reverse=False) -> list:
    """
    - Gives back indices of string interval defined by sub_start and sub_end
    :param string: String to be processed
    :param sub_start: The sub-string where search should start
    :param sub_end: The sub-string where search should end
    :param reverse: Direction of search (going back or forth from start)
    :return: Tuple containing interval, if none is found, a message will be shown
    """
    if reverse:
        string = string[::-1]
        sub_start = sub_start[::-1]
        sub_end = sub_end[::-1]
        rev_correction = len(string)
    else:
        rev_correction = 0

    start_idx = string.find(sub_start)

    len_rest = len(string[start_idx:])
    len_sub_end = len(sub_end)
    for rest_idx in range(len_rest-len_sub_end+1):
        next_letters = ''
        for i in range(len_sub_end):
            next_letters += string[start_idx+rest_idx+i]
        if next_letters == sub_end:
            res = [abs(start_idx - rev_correction), abs(start_idx + rest_idx+len_sub_end - rev_correction)]
            res.sort()
            return res

    print('String interval not found')


def insert_string(original_string: str, insertion: Union[str, List[str]], pos: Union[int, List[int]]) -> str:
    """
    - Inserts sub-string(s) at given position(s)
    :param original_string: Original string
    :param insertion: Sub-string(s) to be inserted
    :param pos: Insertion position or positions, MUST BE GIVEN IN ASCENDING ORDER
    :return: Original string with insertion at pos
    """
    # Check if multiple insertions are given with corresponding positions
    insertion_iter_check = hasattr(insertion, '__iter__')
    pos_iter_check = hasattr(pos, '__iter__')
    if hasattr(insertion, '__iter__') or hasattr(pos, '__iter__'):
        if insertion_iter_check != pos_iter_check:
            raise ValueError(f'One argument is iterable, the other is not!')
        new_string = original_string[:pos[0]] + insertion[0] + original_string[pos[0]:]
        # Previously injected string shifts other indices to right
        right_shift = len(insertion[0])
        for insert_i, pos_i in zip(insertion[1:], pos[1:]):
            new_string = new_string[:pos_i+right_shift] + insert_i + new_string[pos_i+right_shift:]
            # Update right shift
            right_shift += len(insert_i)
    else:
        new_string = original_string[:pos] + insertion + original_string[pos:]
    return new_string


def get_sub_string_data(original_string, sub_start, terminal_ini,
                        terminal_fin):
    """
    - Gets sub-string defined by a starting point and terminal subs in either direction
    :param original_string:
    :param sub_start:
    :param terminal_ini:
    :param terminal_fin:
    :return:
    """
    # Get first and second part of string from starting index
    old_i_st, old_i_en = sub_to_sub(string=original_string,
                                    sub_start=sub_start,
                                    sub_end=terminal_ini,
                                    reverse=True)
    old_ii_st, old_ii_en = sub_to_sub(string=original_string,
                                      sub_start=sub_start,
                                      sub_end=terminal_fin,
                                      reverse=False)
    old_i = original_string[old_i_st:old_i_en]
    old_ii = original_string[old_ii_st:old_ii_en]

    # Create the last layer
    old_input = old_i[:-len(sub_start)] + old_ii

    return old_input, old_ii_en



