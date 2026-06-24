# MIT License
#
# Copyright (c) 2023-2025 Hewlett Packard Enterprise Development LP 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
from datetime import timedelta


def mkdir_p(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)
        return dir
    return False


def convert_nodelist_to_node_nums(nid_str):
    if nid_str == "dummy":
        return -1
    node_nums = []
    
    # There is surely a thick regex to do this that I can't comprehend
    entry_slice_points = [-1]
    in_brackets = False

    for i_char, char in enumerate(nid_str):
        if char == "[":#]
            in_brackets = True
        elif char == "]":
            in_brackets = False
        elif not in_brackets and char == ",":
            entry_slice_points.append(i_char)

    entry_slice_points.append(None)

    for slice_l, slice_r in zip(entry_slice_points[:-1], entry_slice_points[1:]):
        nid_str_entry = nid_str[slice_l+1:slice_r]
        
        # name001
        if "[" not in nid_str_entry:#]
            node_nums.append(nid_str_entry)
            continue

        if nid_str_entry.count("[") >= 2:#]
            raise NotImplementedError(
                "Mulitple numeric ranges like {} not implemented".format(nid_str)
            )

        # name00[1-4,10,12,13-20]
        nid_prefix = nid_str_entry.split("[")[0]#]
        nid_suffix_str = nid_str_entry.strip("]").split("[")[1]#]

        for nid_suffix_entry in nid_suffix_str.split(","):
            if "-" not in nid_suffix_entry:
                node_nums.append(nid_prefix + nid_suffix_entry)
                continue

            nid_suffix_range = nid_suffix_entry.split("-")
            digits = len(nid_suffix_range[0]) if nid_suffix_range[0].startswith("0") else None

            for node_num in range(int(nid_suffix_range[0]), int(nid_suffix_range[1]) + 1):
                node_num = str(node_num)

                if digits is not None:
                    while len(node_num) < digits:
                        node_num = "0" + node_num

                node_nums.append(nid_prefix + node_num)

    return node_nums


def get_sbatch_cli_arg(submit_line, long="", short=""):
    words = submit_line.strip(" ").split(" ")
    target_arg = None
    for i_last_word, word in enumerate(words[1:]):
        # Batch script or executable marks end of options
        if word[0] != "-" and (words[i_last_word][0] != "-" or "=" in words[i_last_word]):
            break
        if long:
            if long + "=" in word:
                target_arg = word.split(long + "=")[1]
                break
            if word == long:
                target_arg = words[i_last_word + 2]
                break
        if short:
            if word == short:
                target_arg = words[i_last_word + 2]
                break

    return target_arg


def timelimit_str_to_timedelta(t_str):
    days, hrs = 0, 0
    try:
        if "-" in t_str:
            days = int(t_str.split("-")[0])
            t_str = t_str.split("-")[1]
    except:
        print(t_str)

    if t_str.count(":") == 1 and t_str.count("."): # MM:SS.SS
        mins, secs = t_str.split(":")
        mins = int(mins)
        secs = float(secs)
    elif t_str.count(":") == 2: ## HH:MM:SS (SS has no decimal place for these ones)
        hrs, mins, secs = map(int, t_str.split(":"))
    else:
        print(t_str)
        raise NotImplementedError("Bruh")

    return timedelta(days=days, hours=hrs, minutes=mins, seconds=secs)


def convert_to_raw(df, cols):
    df[cols] = df[cols].astype(str)
    df[cols] = df[cols].replace(
        { "K" : "e+03", "M" : "e+06", "G" : "e+09", "T" : "e+12" }, regex=True
    ).astype(float).astype(int)
    return df

