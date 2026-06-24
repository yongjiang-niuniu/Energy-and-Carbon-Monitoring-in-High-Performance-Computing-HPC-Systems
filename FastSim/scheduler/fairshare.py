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

import pandas as pd
import numpy as np

# NOTE Assuming that users appear on the at levels where there are only other users (other leaf
# nodes). This is relevant to tie breakers on mixed account user levels


class Root:
    def __init__(self, initial_usage, name="root"):
        self.name = name
        self.usage = initial_usage

        self.new_child_usage = False

        self.children = []

        self.is_root = True
        self.is_leaf = False

    def add_child(self, child):
        self.children.append(child)
        child.add_parent(self)

    def __str__(self):
        ret = "root\n"
        for child in self.children:
            ret += child.__str__(1)
        return ret


# {Account, User} us the unique identifier i think, so a job will point to an account and user
# (which may belong to multiple accounts). This account and user will have there usages updated
# then the usage will be propagated up the tree.
class Account:
    def __init__(self, name, shares, initial_usage):
        self.name = name
        self.shares = shares
        self.usage = initial_usage

        self.new_child_usage = False

        self.children = []
        self.parent = None

        self.is_root = False
        self.is_leaf = False

        self.levelfs = np.inf if shares else 0

    def add_parent(self, parent):
        if self.parent:
            raise ValueError("Already assigned a parent!")
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)
        child.add_parent(self)

    def __str__(self, level=0):
        ret = "\t"*level + self.name + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


# Users with the same name may coexist under different accounts
class User:
    def __init__(self, name, shares, initial_usage, partition, max_jobs, max_submit):
        self.name = name
        self.partition = partition
        self.shares = shares
        self.usage = initial_usage

        self.max_jobs = max_jobs
        self.max_submit = max_submit

        self.parent = None

        self.is_root = False
        self.is_leaf = True

        self.levelfs = np.inf if shares else 0
        self.fairshare_factor = 1.0

    def add_parent(self, parent):
        if self.parent:
            raise ValueError("Already assigned a parent!")
        self.parent = parent

    def __str__(self, level=0):
        ret = "\t"*level + self.name + "\n"
        return ret


class FairTree:
    def __init__(
        self, assoc_file, calc_period, decay_halflife, init_time, active_usrs, excess_usr_assocs,
        partitions
    ):
        self.last_calc_time = init_time
        self.calc_period = calc_period
        # decay constant for 1 second applied for the duration of a calc interval
        self.decay_constant = (1 + np.log(1/2) / decay_halflife.total_seconds())
        self.decay_constant_this_traversal = None

        self.root_node, flat_tree = self._load_tree_slurm(
            assoc_file, active_usrs, excess_usr_assocs, partitions
        )
        self.levels = [[self.root_node]]

        current_level = 0
        while(len(self.levels) > current_level):
            all_level_children = []
            for node in self.levels[current_level]:
                if node.is_leaf:
                    continue
                for child_node in node.children:
                    all_level_children.append(child_node)

            if all_level_children:
                self.levels.append(all_level_children)

            current_level += 1

        self.assocs, self.tot_num_assocs = {}, 0
        for node in flat_tree:
            if not node.is_leaf:
                continue

            if node.partition is None:
                for partition in partitions.partitions:
                    self.assocs[(node.name, partition, node.parent.name)] = node
            else:
                self.assocs[(node.name, node.partition, node.parent.name)] = node

            self.tot_num_assocs += 1

        print("Num unique user assocs = {}".format(self.tot_num_assocs))

    def next_calc(self):
        return self.last_calc_time + self.calc_period

    def job_finish_usage_update(self, job, time):
        # This is just how it's implemented in source code as far as I can tell
        delta_t = max(
            (min(job.endlimit, time) - max(self.last_calc_time, job.start)).total_seconds(), 0
        )
        usage = job.nodes * delta_t ** (self.decay_constant ** delta_t)
        user_node = self.assocs[job.assoc]
        self._update_usages(user_node, usage)

    def fairshare_calc(self, running_jobs, time):
        decay_constant = (
            self.decay_constant ** (time - self.last_calc_time).total_seconds()
        )

        self._decay_all(decay_constant)

        # Collect usages from running jobs between now and last_calc_time
        for job in running_jobs:
            if job.start > self.last_calc_time:
                delta_t = (time - job.start).total_seconds()
                usage = job.nodes * delta_t * (self.decay_constant ** delta_t)
            else:
                delta_t = (time - self.last_calc_time).total_seconds()
                usage = job.nodes * delta_t * decay_constant

            user_node = self.assocs[job.assoc]

            self._update_usages(user_node, usage)

        # Compute levelFS and sort (decay past usages as we go)
        self._tree_traversal(self.root_node)

        self.last_calc_time = time

    def _decay_all(self, decay_constant):
        for nodes in self.levels:
            for node in nodes:
                node.usage *= decay_constant

    # NOTE Ties between sibling users handled properly but not implemented ties between accounts
    # (merge children and sort)
    def _tree_traversal(self, current_node, rank=-1, last_leaf_levelfs=None, tie_cnt=0):
        if current_node.is_leaf:
            if current_node.levelfs == last_leaf_levelfs:
                tie_cnt += 1
            else:
                rank += 1 + tie_cnt
                tie_cnt = 0
                last_leaf_levelfs = current_node.levelfs

            current_node.fairshare_factor = 1.0 - rank / self.tot_num_assocs

            return rank, last_leaf_levelfs, tie_cnt

        if current_node.new_child_usage: # Avoid re-sorting when order is unchanged
            for child_node in current_node.children:
                if child_node.usage:
                    child_node.levelfs = child_node.shares / child_node.usage
                else:
                    child_node.levelfs = np.inf if child_node.shares else 0
            current_node.new_child_usage = False
            current_node.children.sort(key=lambda node: (node.levelfs, node.name), reverse=True)

        for child_node in current_node.children:
            rank, last_leaf_levelfs, tie_cnt = self._tree_traversal(
                child_node, rank=rank, last_leaf_levelfs=last_leaf_levelfs, tie_cnt=tie_cnt
            )

        return rank, last_leaf_levelfs, tie_cnt

    def _update_usages(self, node, usage):
        node.usage += usage
        while not node.is_root:
            node = node.parent
            node.usage += usage
            node.new_child_usage = True

    def _load_tree_slurm(self, assoc_file, active_usrs, excess_usr_assocs, partitions):
        df = pd.read_csv(assoc_file, delimiter='|', lineterminator='\n', header=0)
        df = df.drop([ col for col in df.columns if "Unnamed" in col ], axis=1)

        root_node = Root(0.0)
        level_nodes = [root_node]
        flat_tree = [root_node]
        usr_assocs_removed = 0

        while level_nodes:
            next_level_nodes = []

            for node in level_nodes:
                for _, child_row in df.loc[(df.ParentName == node.name)].iterrows():
                    acc = Account(child_row.Account, 1, 0.0)
                    flat_tree.append(acc)
                    node.add_child(acc)
                    next_level_nodes.append(acc)

                    for _, usr_row in (
                        df.loc[(df.Account == acc.name) & (df.User.notna())].iterrows()
                    ):
                        if (
                            usr_assocs_removed < excess_usr_assocs and
                            usr_row.User not in active_usrs
                        ):
                            usr_assocs_removed += 1
                            continue

                        if pd.isna(usr_row.Partition):
                            partition = None
                        elif usr_row.Partition in partitions.partitions_by_name:
                            partition = partitions.partitions_by_name[usr_row.Partition]
                        else:
                            continue

                        max_jobs = None if pd.isna(usr_row.MaxJobs) else int(usr_row.MaxJobs)
                        max_submit = None if pd.isna(usr_row.MaxSubmit) else int(usr_row.MaxSubmit)

                        usr = User(usr_row.User, 1, 0.0, partition, max_jobs, max_submit)
                        flat_tree.append(usr)
                        acc.add_child(usr)

            level_nodes = next_level_nodes

        return root_node, flat_tree

    def __str__(self):
        ret = (
            "\t".join(
                [ "l{} ({})".format(level, len(nodes)) for level, nodes in enumerate(self.levels) ]
            ) +
            "\n"
        )
        ret += self.levels[0][0].__str__()
        return ret

