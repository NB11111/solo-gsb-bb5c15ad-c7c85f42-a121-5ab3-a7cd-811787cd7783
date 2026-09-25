"""空档素材填充规划核心逻辑（纯标准库实现，不依赖 Web 框架）。

规则概述
--------
* 现有口令沿用 ``POST /compile`` 的口令数组，先用同一个编排器解析并校验每条口令的
  起止毫秒；取目标 ``channel`` 上落（含部分落）在半开区间
  ``[0, horizon_end_ms)`` 内的占用区间，逐段裁剪到该范围内再求并集，
  其余部分即空档（半开、长度为正），按起点升序排列，至多 8 个。
* 每条素材（filler）含唯一 ``id``、正整数 ``duration_ms``、非负整数 ``priority``，
  至多 12 条；每条素材**最多使用一次**，可在全部空档之间联合分配，也可不使用。
  素材时长之和不得超过所在空档长度，素材之间**不拆分、不重复**。
* 目标依次为：① **填充总时长最大**（入选素材时长之和）；② 并列时
  **priority 总和最大**；③ 仍并列时，取「按空档时间顺序、各空档内素材 id 升序
  紧贴排布后展开的素材 id 序列」**字典序最小**的方案。
* 每个空档内入选素材按 id 升序，从空档起点开始紧贴排布；返回每条入选素材的
  起止时刻，以及每个空档剩余未填充的时长。

求解方法（素材至多 12 条）

* 空档容量是子集和约束：全集只有 2^12 = 4096 个子集，单空档枚举「本空档子集 +
  剩余池」对至多 3^12 ≈ 53 万次，8 个空档常数极小，无需逐毫秒或逐方案枚举。
* 反向子集 DP 逐空档求「剩余素材池在当前及之后空档」的最优记录
  （填充时长、priority、展开序列编码），并记录本空档选取的子集供重建。
* 展开 id 序列编码为 n 位 big-endian 位置制整数（数字 = 素材在 id 升序中的
  下标 + 1，未用位置为 0），数值比较与「右侧补 0 等长元组」逐位比较一致，
  即等价于不同长度元组的字典序比较；故第三级裁决退化为一次整数比较。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from .compiler import CompileError, compile_commands

# 素材条数上限与空档数量上限
MAX_FILLERS = 12
MAX_GAPS = 8


class FillerPlanError(CompileError):
    """空档填充规划失败的业务异常；复用 CompileError 的 422 转换通道。"""


def plan_fillers(
    commands: list[dict[str, Any]],
    channel: str,
    horizon_end_ms: int,
    fillers: list[dict[str, Any]],
) -> dict[str, Any]:
    """在目标通道 ``[0, horizon_end_ms)`` 的空档间联合分配素材，求最优填充方案。

    不会修改传入的 ``commands`` / ``fillers``。

    :raises CompileError: 口令编排非法（重复 id、依赖环、负起点等，与 /compile 相同）。
    :raises FillerPlanError: 通道为空、horizon 非正、素材非法或空档超过上限时抛出，
        ``code`` 可用于程序化区分错误类型。
    """
    # 1) 复用编排器解析并校验现有口令（深拷贝，setdefault 等不会回写调用方数据；
    #    填充规划只需解析后的时间线，跳过 O(n²) 的冲突对枚举）
    compiled = compile_commands(deepcopy(commands), with_conflicts=False)
    timeline = compiled["timeline"]

    # 2) 参数与素材校验
    if channel == "":
        raise FillerPlanError(
            "INVALID_CHANNEL", "填充目标通道名不能为空", {"channel": channel}
        )
    if not isinstance(horizon_end_ms, int) or isinstance(horizon_end_ms, bool) or horizon_end_ms <= 0:
        raise FillerPlanError(
            "INVALID_REQUEST",
            "horizon_end_ms 必须为正整数",
            {"horizon_end_ms": horizon_end_ms},
        )
    _validate_fillers(fillers)

    # 3) 裁剪并求并集，得到至多 8 个半开空档
    gaps = _collect_gaps(timeline, channel, horizon_end_ms)

    # 4) 素材按 id 升序编号；时长超过所有空档长度的素材不可能入选
    filler_items = sorted(fillers, key=lambda item: item["id"])
    ids = [item["id"] for item in filler_items]
    durations = [item["duration_ms"] for item in filler_items]
    priorities = [item["priority"] for item in filler_items]

    # 5) 子集 DP + 可行性记忆化，选最优使用集合并按字典序最小重建分组
    used_mask, group_masks = _allocate(gaps, durations, priorities)

    # 6) 按空档时间顺序生成紧贴排布的入选素材时刻，并统计各空档剩余时长
    placements: list[dict[str, Any]] = []
    gap_results: list[dict[str, Any]] = []
    for gap, group_mask in zip(gaps, group_masks):
        gap_start, gap_end, gap_length = gap
        cursor = gap_start
        group_filled = 0
        filler_ids: list[str] = []
        for i, filler_id in enumerate(ids):
            if group_mask & (1 << i):
                duration = durations[i]
                placements.append(
                    {
                        "filler_id": filler_id,
                        "start_ms": cursor,
                        "end_ms": cursor + duration,
                    }
                )
                filler_ids.append(filler_id)
                cursor += duration
                group_filled += duration
        gap_results.append(
            {
                "start_ms": gap_start,
                "end_ms": gap_end,
                "remaining_ms": gap_length - group_filled,
                "filler_ids": filler_ids,
            }
        )

    selected_ids = [
        ids[i] for i in range(len(filler_items)) if used_mask & (1 << i)
    ]
    total_filled_ms = sum(
        durations[i] for i in range(len(filler_items)) if used_mask & (1 << i)
    )
    total_priority = sum(
        priorities[i] for i in range(len(filler_items)) if used_mask & (1 << i)
    )
    return {
        "channel": channel,
        "horizon_end_ms": horizon_end_ms,
        "gaps": gap_results,
        "placements": placements,
        "selected_filler_ids": selected_ids,
        "total_filled_ms": total_filled_ms,
        "total_priority": total_priority,
    }


def _validate_fillers(fillers: list[dict[str, Any]]) -> None:
    """按既定顺序校验素材数组。"""
    # 1) 空 id
    empty_ids = [item["id"] for item in fillers if item["id"] == ""]
    if empty_ids:
        raise FillerPlanError(
            "INVALID_ID",
            "素材 id 不能为空",
            {"ids": empty_ids},
        )

    # 2) 重复 id
    seen: set[str] = set()
    duplicates: list[str] = []
    for item in fillers:
        fid = item["id"]
        if fid in seen:
            if fid not in duplicates:
                duplicates.append(fid)
            continue
        seen.add(fid)
    if duplicates:
        raise FillerPlanError(
            "DUPLICATE_ID",
            f"存在重复的素材 id: {', '.join(sorted(duplicates))}",
            {"duplicates": sorted(duplicates)},
        )

    # 3) 至多 12 条
    if len(fillers) > MAX_FILLERS:
        raise FillerPlanError(
            "TOO_MANY_FILLERS",
            f"素材至多 {MAX_FILLERS} 条，当前提供了 {len(fillers)} 条",
            {"count": len(fillers), "limit": MAX_FILLERS},
        )

    # 4) duration_ms 必须为正整数
    bad_duration = [
        item["id"]
        for item in fillers
        if not isinstance(item["duration_ms"], int)
        or isinstance(item["duration_ms"], bool)
        or item["duration_ms"] <= 0
    ]
    if bad_duration:
        raise FillerPlanError(
            "INVALID_DURATION",
            "素材 duration_ms 必须为正整数",
            {"ids": bad_duration},
        )

    # 5) priority 必须为非负整数
    bad_priority = [
        item["id"]
        for item in fillers
        if not isinstance(item["priority"], int)
        or isinstance(item["priority"], bool)
        or item["priority"] < 0
    ]
    if bad_priority:
        raise FillerPlanError(
            "INVALID_REQUEST",
            "素材 priority 必须为非负整数",
            {"ids": bad_priority},
        )


def _collect_gaps(
    timeline: list[dict[str, Any]], channel: str, horizon_end_ms: int
) -> list[tuple[int, int, int]]:
    """裁剪目标通道口令区间到 ``[0, horizon_end_ms)``、求并集，返回半开空档。

    :returns: 每个空档为 ``(起点, 终点, 长度)``，按起点升序；只包含长度为正的空档。
    """
    clipped: list[tuple[int, int]] = []
    for entry in timeline:
        if entry["channel"] != channel:
            continue
        start = max(entry["start_ms"], 0)
        end = min(entry["end_ms"], horizon_end_ms)
        if start < end:  # 半开区间，长度为正才构成占用
            clipped.append((start, end))

    clipped.sort()
    merged: list[tuple[int, int]] = []
    for start, end in clipped:
        if merged and start <= merged[-1][1]:
            # 半开区间：首尾相接（start == 上一段 end）也可并入同一段占用
            if end > merged[-1][1]:
                merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    gaps: list[tuple[int, int, int]] = []
    cursor = 0
    for start, end in merged:
        if cursor < start:
            gaps.append((cursor, start, start - cursor))
        cursor = max(cursor, end)
    if cursor < horizon_end_ms:
        gaps.append((cursor, horizon_end_ms, horizon_end_ms - cursor))

    if len(gaps) > MAX_GAPS:
        raise FillerPlanError(
            "TOO_MANY_GAPS",
            f"通道 {channel} 在 [0, {horizon_end_ms}) 内有 {len(gaps)} 个空档，"
            f"至多允许 {MAX_GAPS} 个",
            {
                "channel": channel,
                "horizon_end_ms": horizon_end_ms,
                "count": len(gaps),
                "limit": MAX_GAPS,
                "gaps": [[start, end] for start, end, _ in gaps],
            },
        )
    return gaps


def _allocate(
    gaps: list[tuple[int, int, int]],
    durations: list[int],
    priorities: list[int],
) -> tuple[int, list[int]]:
    """在全部空档间联合分配素材，返回 ``(使用集合掩码, 各空档分组掩码列表)``。

    目标：填充总时长最大 → priority 总和最大 → 展开 id 序列字典序最小。
    素材已按 id 升序编号，故下标顺序即 id 顺序。

    展开序列固定占 ``n`` 个位置（右侧以 0 补齐），用一个 ``n`` 位的位置制整数
    编码：第 ``k`` 个展开位置（k 从 0 起，越早空档 / 空档内越靠前越小）对应
    ``base^(n-1-k)`` 位，数字为「素材下标 + 1」（未使用位置为 0）。于是整数的
    数值比较与「右侧补 0 后的等长元组」逐位比较完全一致，而这正等价于 Python
    对不同长度元组的字典序比较（短序列是长序列前缀时短者更小）。

    反向子集 DP：每行给出「剩余素材池 mask 在当前及之后空档」的最优记录
    ``(填充时长, priority, 序列编码, 本空档子集, 剩余池)``。
    """
    n = len(durations)
    size = 1 << n
    full = size - 1
    base = n + 1

    # 子集时长 / priority 总和（递推，避免重复求和）
    sub_duration = [0] * size
    sub_priority = [0] * size
    for mask in range(1, size):
        low = mask & -mask
        i = low.bit_length() - 1
        rest = mask ^ low
        sub_duration[mask] = sub_duration[rest] + durations[i]
        sub_priority[mask] = sub_priority[rest] + priorities[i]

    # 各空档下「子集总时长不超过容量」的布尔表
    fit_table = [
        [sub_duration[mask] <= length for mask in range(size)]
        for _, _, length in gaps
    ]
    gap_count = len(gaps)

    # place_value[k]：第 k 个展开位置（自 0 起）对应的位权 base^(n-1-k)
    place_value = [base ** (n - 1 - k) for k in range(n)]
    # base_pow[k] = base^k，用于把后续序列向右移动 k 个展开位置
    base_pow = [1] * (n + 1)
    for k in range(1, n + 1):
        base_pow[k] = base_pow[k - 1] * base
    # sub_code[mask]：子集按 id 升序占据前 popcount(mask) 个展开位置时的编码。
    # 最低位 i 是子集内最小下标，应排在最前（最高位 pv[0]）；其余下标整体右移一位，
    # 即把 rest 的编码除以 base（rest 末尾至少有一个 0 填充位，整除精确）。
    sub_code = [0] * size
    for mask in range(1, size):
        low = mask & -mask
        i = low.bit_length() - 1
        rest = mask ^ low
        sub_code[mask] = (i + 1) * place_value[0] + sub_code[rest] // base

    # 反向 DP。每条记录为
    # (填充时长, priority, n 位序列编码, 本空档子集, 剩余池)。
    # 序列编码自第 0 个展开位置（最高位）起放置入选素材的「下标 + 1」，未用位置为 0。
    # 边界：没有空档时任何素材都不使用，序列编码为 0。
    # dp_rows[g] 对应第 g 个空档（dp_rows[gap_count] 为边界行），自后向前填充；
    # 每个 mask 都至少有「本空档什么都不放（sub=0）」的可行转移，故行内不会有缺值。
    row_type = tuple[int, int, int, int, int]
    dp_rows: list[list[row_type]] = [
        [(0, 0, 0, 0, mask) for mask in range(size)]
    ]
    for gap_index in range(gap_count - 1, -1, -1):
        fit = fit_table[gap_index]
        next_row = dp_rows[0]
        row: list[row_type] = [(0, 0, 0, 0, 0)] * size
        for mask in range(size):
            best: row_type | None = None
            sub = mask
            while True:
                if fit[sub]:
                    rest_record = next_row[mask ^ sub]
                    (
                        rest_filled,
                        rest_priority,
                        rest_seq,
                        _,
                        _,
                    ) = rest_record
                    current_count = sub.bit_count()
                    # 拼接：本空档升序序列占据前 current_count 位（保持高位不变），
                    # 后续空档序列整体向右（低位方向）移动 current_count 位；
                    # sub 与 rest 素材互不相交，故被移出的 current_count 位必为 0，
                    # 整除精确。例：sub=[c]、rest=[e] -> 编码 [c, e, 0, ...]。
                    sequence_code = (
                        sub_code[sub]
                        + rest_seq // base_pow[current_count]
                    )
                    candidate = (
                        sub_duration[sub] + rest_filled,
                        sub_priority[sub] + rest_priority,
                        sequence_code,
                        sub,
                        mask ^ sub,
                    )
                    # 比较：时长降序、priority 降序、序列编码升序
                    if best is None or (
                        candidate[0] > best[0]
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
                if sub == 0:
                    break
                sub = (sub - 1) & mask
            assert best is not None
            row[mask] = best
        dp_rows.insert(0, row)

    # 沿「本空档子集 / 剩余池」指针重建各空档分组（dp_rows[0][full] 即全局最优）。
    # pool 始终是尚未决定归属的可用素材池；未被任何空档选中的素材最后留在 pool 中，
    # 因此使用集合 = 各空档分组之并。
    group_masks: list[int] = []
    used_mask = 0
    pool = full
    for gap_index in range(gap_count):
        _, _, _, chosen, next_pool = dp_rows[gap_index][pool]
        group_masks.append(chosen)
        used_mask |= chosen
        pool = next_pool

    return used_mask, group_masks
