"""app.fillers 的单元测试（仅依赖标准库，可直接 `python -m unittest discover`）。"""

from __future__ import annotations

import itertools
import random
import unittest

from app.compiler import CompileError, START
from app.fillers import MAX_FILLERS, MAX_GAPS, FillerPlanError, plan_fillers


def cmd(cid, channel, anchor=START, offset=0, duration=1000):
    return {
        "id": cid,
        "channel": channel,
        "anchor": anchor,
        "offset_ms": offset,
        "duration_ms": duration,
    }


def filler(fid, duration, priority=0):
    return {"id": fid, "duration_ms": duration, "priority": priority}


def gap_commands(gaps, channel="main", horizon=None):
    """按给定半开空档列表反推同通道互不重叠的占用口令。

    ``gaps`` 为 (起点, 终点) 列表；空档之外全部放同通道占位口令。
    """
    commands = []
    cursor = 0
    for index, (start, end) in enumerate(gaps):
        if cursor < start:
            commands.append(
                cmd(
                    f"occ{index}",
                    channel,
                    START,
                    cursor,
                    start - cursor,
                )
            )
        cursor = end
    end = horizon if horizon is not None else cursor
    if cursor < end:
        commands.append(
            cmd("occlast", channel, START, cursor, end - cursor)
        )
    return commands, end


def flatten_ids(result):
    sequence = []
    for gap in result["gaps"]:
        sequence.extend(gap["filler_ids"])
    return sequence


class GapComputationTests(unittest.TestCase):
    def test_empty_channel_whole_horizon_is_one_gap(self):
        result = plan_fillers([], "main", 1000, [])
        self.assertEqual(len(result["gaps"]), 1)
        self.assertEqual(
            (
                result["gaps"][0]["start_ms"],
                result["gaps"][0]["end_ms"],
                result["gaps"][0]["remaining_ms"],
            ),
            (0, 1000, 1000),
        )
        self.assertEqual(result["placements"], [])
        self.assertEqual(result["selected_filler_ids"], [])
        self.assertEqual(result["total_filled_ms"], 0)
        self.assertEqual(result["total_priority"], 0)

    def test_other_channels_ignored(self):
        commands = [cmd("a", "sub", START, 0, 1000)]
        result = plan_fillers(commands, "main", 1000, [])
        self.assertEqual(len(result["gaps"]), 1)
        self.assertEqual(
            (result["gaps"][0]["start_ms"], result["gaps"][0]["end_ms"]),
            (0, 1000),
        )

    def test_clip_intervals_to_horizon_and_merge_touching(self):
        # main: [0,100) 与 [100,200) 首尾相接，裁剪到 horizon=150 后并为 [0,150)
        commands = [
            cmd("a", "main", START, 0, 100),
            cmd("b", "main", START, 100, 100),
        ]
        result = plan_fillers(commands, "main", 150, [])
        # 占用覆盖整个 [0,150)：没有长度为正的空档
        self.assertEqual(result["gaps"], [])
        self.assertEqual(result["total_filled_ms"], 0)

    def test_partial_overlap_is_clipped(self):
        # 口令 [80,180)，horizon=120：裁剪为占用 [80,120)，空档 [0,80) 与 [120,120) 零长
        commands = [cmd("a", "main", START, 80, 100)]
        result = plan_fillers(commands, "main", 120, [])
        self.assertEqual(len(result["gaps"]), 1)
        self.assertEqual(
            (result["gaps"][0]["start_ms"], result["gaps"][0]["end_ms"]),
            (0, 80),
        )

    def test_overlapping_commands_merge_to_one_occupied_run(self):
        commands = [
            cmd("a", "main", START, 0, 100),
            cmd("b", "main", START, 50, 100),
        ]
        result = plan_fillers(commands, "main", 200, [])
        self.assertEqual(len(result["gaps"]), 1)
        self.assertEqual(
            (result["gaps"][0]["start_ms"], result["gaps"][0]["end_ms"]),
            (150, 200),
        )

    def test_multiple_gaps_in_time_order(self):
        gaps = [(0, 100), (200, 300), (400, 500)]
        commands, horizon = gap_commands(gaps, horizon=600)
        result = plan_fillers(commands, "main", horizon, [])
        actual = [(g["start_ms"], g["end_ms"]) for g in result["gaps"]]
        self.assertEqual(actual, gaps)

    def test_more_than_eight_gaps_raises(self):
        # 9 个互不相接的占用区间 -> 10 个空档
        commands = [
            cmd(f"c{i}", "main", START, i * 20 + 10, 5) for i in range(9)
        ]
        with self.assertRaises(FillerPlanError) as ctx:
            plan_fillers(commands, "main", 9 * 20 + 5, [])
        self.assertEqual(ctx.exception.code, "TOO_MANY_GAPS")
        self.assertEqual(ctx.exception.details["count"], 10)
        self.assertEqual(ctx.exception.details["limit"], MAX_GAPS)

    def test_exactly_eight_gaps_allowed(self):
        # 7 个占用段 -> 8 个空档
        commands = [cmd(f"c{i}", "main", START, i * 20 + 10, 5) for i in range(7)]
        result = plan_fillers(commands, "main", 7 * 20 + 5, [])
        self.assertEqual(len(result["gaps"]), 8)


class AllocationBasicTests(unittest.TestCase):
    def test_single_gap_tight_pack_by_id_ascending(self):
        commands, horizon = gap_commands([(0, 1000)])
        result = plan_fillers(
            commands,
            "main",
            horizon,
            [filler("z", 100), filler("a", 200), filler("m", 300)],
        )
        # 总时长 600 全部放入；空档内 id 升序紧贴
        self.assertEqual(result["gaps"][0]["filler_ids"], ["a", "m", "z"])
        self.assertEqual(
            [(p["filler_id"], p["start_ms"], p["end_ms"]) for p in result["placements"]],
            [("a", 0, 200), ("m", 200, 500), ("z", 500, 600)],
        )
        self.assertEqual(result["gaps"][0]["remaining_ms"], 400)
        self.assertEqual(result["total_filled_ms"], 600)
        self.assertEqual(result["selected_filler_ids"], ["a", "m", "z"])

    def test_capacity_subset_sum_chooses_max_duration(self):
        # 空档长度 10；素材 6 / 5 / 5：最大填充为 5+5=10
        commands, horizon = gap_commands([(0, 10)])
        result = plan_fillers(
            commands,
            "main",
            horizon,
            [filler("a", 6), filler("b", 5), filler("c", 5)],
        )
        self.assertEqual(result["total_filled_ms"], 10)
        self.assertEqual(flatten_ids(result), ["b", "c"])

    def test_joint_allocation_across_gaps(self):
        # 空档长度 4 与 6；素材 3 / 3 / 4：4 必须独立占长度 4 的空档，
        # 两个 3 合占长度 6 -> 填满 10
        gaps = [(0, 4), (20, 26)]
        commands, horizon = gap_commands(gaps)
        result = plan_fillers(
            commands,
            "main",
            horizon,
            [filler("a", 3), filler("b", 3), filler("c", 4)],
        )
        self.assertEqual(result["total_filled_ms"], 10)
        groups = [g["filler_ids"] for g in result["gaps"]]
        self.assertEqual(groups, [["c"], ["a", "b"]])

    def test_each_filler_used_at_most_once(self):
        gaps = [(0, 100), (200, 300)]
        commands, horizon = gap_commands(gaps)
        result = plan_fillers(
            commands, "main", horizon, [filler("a", 50), filler("b", 50)]
        )
        self.assertEqual(flatten_ids(result).count("a"), 1)
        self.assertEqual(flatten_ids(result).count("b"), 1)

    def test_priority_breaks_duration_tie(self):
        # 空档长度 5：a(3,p=9) + b(2,p=0) 与 c(5,p=1) 都填满 5；
        # 优先 priority 总和：a+b=9 > c=1
        commands, horizon = gap_commands([(0, 5)])
        result = plan_fillers(
            commands,
            "main",
            horizon,
            [
                filler("a", 3, 9),
                filler("b", 2, 0),
                filler("c", 5, 1),
            ],
        )
        self.assertEqual(result["total_filled_ms"], 5)
        self.assertEqual(result["total_priority"], 9)
        self.assertEqual(flatten_ids(result), ["a", "b"])

    def test_lex_sequence_breaks_final_tie(self):
        # 空档长度 5：a(2)+c(3) 与 b(5) 都填满且 priority 相同；
        # 展开序列 ['a','c'] 字典序小于 ['b']
        commands, horizon = gap_commands([(0, 5)])
        result = plan_fillers(
            commands,
            "main",
            horizon,
            [filler("a", 2, 0), filler("b", 5, 0), filler("c", 3, 0)],
        )
        self.assertEqual(flatten_ids(result), ["a", "c"])

    def test_lex_tie_across_gaps(self):
        # 档1 容量 2、档2 容量 3，素材 a(3)、b(2)。两种填满方式：
        # ① b 进档1、a 进档2 -> seq ['b','a']；不存在其它可行排列。
        # 再加素材 c(3)：a 与 c 都只能进档2，取 id 更小的 a。
        gaps = [(0, 2), (10, 13)]
        commands, horizon = gap_commands(gaps)
        result = plan_fillers(
            commands,
            "main",
            horizon,
            [filler("a", 3, 0), filler("b", 2, 0), filler("c", 3, 0)],
        )
        self.assertEqual(result["total_filled_ms"], 5)
        self.assertEqual(flatten_ids(result), ["b", "a"])

    def test_lex_tie_with_shorter_sequence_prefix(self):
        # 只有一个空档（容量 3），素材 a(2)、b(1)、c(3)。
        # 方案① c 单独填满     -> seq ['c']，     填充 3
        # 方案② a+b 一起填满  -> seq ['a','b']，填充 3
        # 填充与 priority 都相同；['a','b'] < ['c']，取方案②。
        gaps = [(0, 3)]
        commands, horizon = gap_commands(gaps)
        result = plan_fillers(
            commands,
            "main",
            horizon,
            [filler("a", 2, 0), filler("b", 1, 0), filler("c", 3, 0)],
        )
        self.assertEqual(result["total_filled_ms"], 3)
        self.assertEqual(result["total_priority"], 0)
        self.assertEqual(flatten_ids(result), ["a", "b"])

    def test_empty_gap_in_middle_still_time_ordered(self):
        # 中间空档容量 1，装不下任何素材；档1 容量 2、档3 容量 50：
        # a(50) 只能进档3，b(2) 只能进档1；序列按空档时间顺序为 ['b','a']。
        gaps = [(0, 2), (20, 21), (40, 90)]
        commands, horizon = gap_commands(gaps)
        result = plan_fillers(
            commands,
            "main",
            horizon,
            [filler("a", 50), filler("b", 2)],
        )
        groups = [g["filler_ids"] for g in result["gaps"]]
        self.assertEqual(groups, [["b"], [], ["a"]])
        self.assertEqual(flatten_ids(result), ["b", "a"])
        placements = [
            (p["filler_id"], p["start_ms"]) for p in result["placements"]
        ]
        self.assertEqual(placements, [("b", 0), ("a", 40)])
        self.assertEqual(result["gaps"][0]["remaining_ms"], 0)
        self.assertEqual(result["gaps"][1]["remaining_ms"], 1)
        self.assertEqual(result["gaps"][2]["remaining_ms"], 0)

    def test_oversized_filler_never_selected(self):
        commands, horizon = gap_commands([(0, 10)])
        result = plan_fillers(
            commands, "main", horizon, [filler("big", 11), filler("ok", 5)]
        )
        self.assertEqual(result["selected_filler_ids"], ["ok"])
        self.assertEqual(result["total_filled_ms"], 5)

    def test_no_fillers_returns_empty_plan(self):
        commands, horizon = gap_commands([(0, 100), (200, 300)])
        result = plan_fillers(commands, "main", horizon, [])
        self.assertEqual(result["placements"], [])
        self.assertEqual(result["selected_filler_ids"], [])
        self.assertEqual([g["remaining_ms"] for g in result["gaps"]], [100, 100])


class PlacementInvariantTests(unittest.TestCase):
    def test_placements_tight_from_gap_start_and_remaining_consistent(self):
        gaps = [(0, 30), (100, 180), (300, 340)]
        commands, horizon = gap_commands(gaps)
        fillers = [filler(chr(97 + i), d, i) for i, d in enumerate([20, 30, 40, 70])]
        result = plan_fillers(commands, "main", horizon, fillers)
        for gap in result["gaps"]:
            cursor = gap["start_ms"]
            used = 0
            for fid in gap["filler_ids"]:
                duration = next(f["duration_ms"] for f in fillers if f["id"] == fid)
                placement = next(
                    p for p in result["placements"] if p["filler_id"] == fid
                )
                self.assertEqual(
                    (placement["start_ms"], placement["end_ms"]),
                    (cursor, cursor + duration),
                )
                cursor += duration
                used += duration
            self.assertEqual(gap["remaining_ms"], gap["end_ms"] - gap["start_ms"] - used)
        self.assertEqual(result["selected_filler_ids"], sorted(result["selected_filler_ids"]))
        # 每条素材最多出现在一个空档
        flattened = flatten_ids(result)
        self.assertEqual(len(flattened), len(set(flattened)))


class ValidationTests(unittest.TestCase):
    def test_empty_channel(self):
        with self.assertRaises(FillerPlanError) as ctx:
            plan_fillers([], "", 100, [])
        self.assertEqual(ctx.exception.code, "INVALID_CHANNEL")

    def test_non_positive_horizon(self):
        for horizon in (0, -1):
            with self.assertRaises(FillerPlanError) as ctx:
                plan_fillers([], "main", horizon, [])
            self.assertEqual(ctx.exception.code, "INVALID_REQUEST")

    def test_duplicate_filler_id(self):
        with self.assertRaises(FillerPlanError) as ctx:
            plan_fillers(
                [], "main", 100, [filler("a", 1), filler("a", 2)]
            )
        self.assertEqual(ctx.exception.code, "DUPLICATE_ID")
        self.assertEqual(ctx.exception.details["duplicates"], ["a"])

    def test_empty_filler_id(self):
        with self.assertRaises(FillerPlanError) as ctx:
            plan_fillers([], "main", 100, [filler("", 1)])
        self.assertEqual(ctx.exception.code, "INVALID_ID")

    def test_invalid_duration(self):
        with self.assertRaises(FillerPlanError) as ctx:
            plan_fillers([], "main", 100, [filler("a", 0)])
        self.assertEqual(ctx.exception.code, "INVALID_DURATION")
        with self.assertRaises(FillerPlanError) as ctx:
            plan_fillers([], "main", 100, [filler("a", -5)])
        self.assertEqual(ctx.exception.code, "INVALID_DURATION")

    def test_negative_priority(self):
        with self.assertRaises(FillerPlanError) as ctx:
            plan_fillers([], "main", 100, [filler("a", 1, -1)])
        self.assertEqual(ctx.exception.code, "INVALID_REQUEST")

    def test_too_many_fillers_in_core(self):
        fillers = [filler(f"f{i}", 1) for i in range(MAX_FILLERS + 1)]
        with self.assertRaises(FillerPlanError) as ctx:
            plan_fillers([], "main", 10**9, fillers)
        self.assertEqual(ctx.exception.code, "TOO_MANY_FILLERS")
        self.assertEqual(ctx.exception.details["count"], MAX_FILLERS + 1)
        self.assertEqual(ctx.exception.details["limit"], MAX_FILLERS)

    def test_exactly_twelve_fillers_allowed(self):
        fillers = [filler(f"f{i:02d}", 1) for i in range(MAX_FILLERS)]
        result = plan_fillers([], "main", 100, fillers)
        self.assertEqual(result["total_filled_ms"], MAX_FILLERS)

    def test_command_compile_errors_propagate(self):
        # 缺失锚点：复用编排解析，错误码与 /compile 一致
        commands = [cmd("a", "main", "ghost", 0, 10)]
        with self.assertRaises(CompileError) as ctx:
            plan_fillers(commands, "main", 100, [])
        self.assertEqual(ctx.exception.code, "MISSING_ANCHOR")

    def test_does_not_mutate_inputs(self):
        commands = [cmd("a", "main", START, 0, 10)]
        fillers = [filler("a", 1)]
        commands_copy = [dict(c) for c in commands]
        fillers_copy = [dict(f) for f in fillers]
        plan_fillers(commands, "main", 100, fillers)
        self.assertEqual(commands, commands_copy)
        self.assertEqual(fillers, fillers_copy)


class BruteForceFuzzTests(unittest.TestCase):
    """小规模随机用例对照逐方案暴力枚举（含全部三级裁决）。"""

    def test_matches_brute_force(self):
        random.seed(20260925)
        for _ in range(300):
            gap_count = random.randint(1, 3)
            capacities = [random.randint(1, 9) for _ in range(gap_count)]
            n = random.randint(0, 6)
            durations = [random.randint(1, 6) for _ in range(n)]
            priorities = [random.randint(0, 3) for _ in range(n)]
            ids = [chr(97 + i) for i in range(n)]

            gap_specs = []
            position = 0
            for capacity in capacities:
                start = position + 10
                gap_specs.append((start, start + capacity))
                position = start + capacity + 10
            horizon = position + 5
            commands, _ = gap_commands(gap_specs, horizon=horizon)
            fillers = [
                filler(ids[i], durations[i], priorities[i]) for i in range(n)
            ]

            result = plan_fillers(commands, "main", horizon, fillers)
            sequence = flatten_ids(result)
            total_duration = sum(
                f["duration_ms"] for f in fillers if f["id"] in result["selected_filler_ids"]
            )
            total_priority = sum(
                f["priority"] for f in fillers if f["id"] in result["selected_filler_ids"]
            )

            expected = self._brute(
                durations, capacities, priorities, ids
            )
            self.assertEqual(
                (total_duration, total_priority, tuple(sequence)),
                expected,
                msg=f"caps={capacities} d={durations} p={priorities}",
            )

            # 空档容量与紧贴排布不变量
            for gap in result["gaps"]:
                filled = sum(
                    f["duration_ms"]
                    for f in fillers
                    if f["id"] in gap["filler_ids"]
                )
                self.assertLessEqual(
                    filled, gap["end_ms"] - gap["start_ms"]
                )
                self.assertEqual(
                    gap["remaining_ms"],
                    gap["end_ms"] - gap["start_ms"] - filled,
                )

    @staticmethod
    def _brute(durations, capacities, priorities, ids):
        gap_count = len(capacities)
        n = len(durations)
        best = None
        # 每条素材分配到某个空档（0..G-1）或不使用（G）
        for assignment in itertools.product(range(gap_count + 1), repeat=n):
            loads = [0] * gap_count
            for i, target in enumerate(assignment):
                if target < gap_count:
                    loads[target] += durations[i]
            if any(loads[g] > capacities[g] for g in range(gap_count)):
                continue
            used = [i for i in range(n) if assignment[i] < gap_count]
            total_duration = sum(durations[i] for i in used)
            total_priority = sum(priorities[i] for i in used)
            sequence = []
            for g in range(gap_count):
                sequence.extend(
                    sorted(ids[i] for i in used if assignment[i] == g)
                )
            candidate = (total_duration, total_priority, tuple(sequence))
            if (
                best is None
                or candidate[0] > best[0]
                or (
                    candidate[0] == best[0]
                    and (
                        candidate[1] > best[1]
                        or (
                            candidate[1] == best[1]
                            and candidate[2] < best[2]
                        )
                    )
                )
            ):
                best = candidate
        return best


if __name__ == "__main__":
    unittest.main()
