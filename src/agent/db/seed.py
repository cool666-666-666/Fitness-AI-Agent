"""在应用启动时向 exercise_catalog 表填充常用健身动作数据。

该脚本在 init_db（建表）之后执行。采用幂等设计——已存在的动作条目会被跳过，
因此重复启动不会产生重复数据。

动作覆盖 7 个主要肌群分类，每个动作包含：标准中文名、肌群分类、英文/中文别名列表。
"""

from __future__ import annotations

from loguru import logger
from sqlalchemy import select

from agent.db.models import ExerciseCatalog
from agent.db.mysql import async_session_factory

# 每个条目格式: (标准中文名称, 肌群分类, [别名列表])
# 别名统一使用小写，便于精确匹配查询。
# 别名的作用：用户输入 "bench press" 时能匹配到 "卧推"，输入 "squat" 时能匹配到 "深蹲"。
_EXERCISES = [
    # ── 胸 ──
    ("卧推", "胸", ["bench press", "barbell bench press"]),
    ("哑铃卧推", "胸", ["dumbbell bench press", "db bench press"]),
    ("上斜卧推", "胸", ["incline bench press", "incline barbell bench"]),
    ("上斜哑铃卧推", "胸", ["incline dumbbell bench press", "incline db bench"]),
    ("下斜卧推", "胸", ["decline bench press"]),
    ("飞鸟", "胸", ["fly", "dumbbell fly", "chest fly", "哑铃飞鸟"]),
    ("绳索夹胸", "胸", ["cable crossover", "cable fly", "龙门架夹胸"]),
    ("双杠臂屈伸", "胸", ["dip", "dips", "臂屈伸"]),
    ("俯卧撑", "胸", ["push up", "pushup", "push-ups"]),
    ("器械推胸", "胸", ["chest press machine", "坐姿推胸"]),

    # ── 背 ──
    ("引体向上", "背", ["pull up", "pullup", "chin up", "chinup"]),
    ("杠铃划船", "背", ["barbell row", "划船"]),
    ("哑铃划船", "背", ["dumbbell row", "db row", "单臂哑铃划船"]),
    ("高位下拉", "背", ["lat pulldown", "pull down", "lat pull down"]),
    ("坐姿划船", "背", ["seated row", "seated cable row", "绳索划船"]),
    ("T杠划船", "背", ["t-bar row", "t bar row"]),
    ("直臂下压", "背", ["straight arm pulldown", "直臂下拉"]),
    ("硬拉", "背", ["deadlift", "传统硬拉", "conventional deadlift"]),
    ("架上硬拉", "背", ["rack pull"]),
    ("对握引体", "背", ["neutral grip pull up"]),

    # ── 腿 ──
    ("深蹲", "腿", ["squat", "杠铃深蹲", "barbell squat", "back squat"]),
    ("前蹲", "腿", ["front squat"]),
    ("腿举", "腿", ["leg press", "倒蹬"]),
    ("哈克深蹲", "腿", ["hack squat"]),
    ("腿屈伸", "腿", ["leg extension"]),
    ("腿弯举", "腿", ["leg curl", "俯卧腿弯举"]),
    ("弓步", "腿", ["lunge", "箭步蹲", "弓步蹲", "walking lunge"]),
    ("保加利亚分腿蹲", "腿", ["bulgarian split squat", "保加利亚蹲"]),
    ("罗马尼亚硬拉", "腿", ["romanian deadlift", "rdl", "直腿硬拉"]),
    ("臀推", "腿", ["hip thrust", "臀桥", "glute bridge"]),
    ("高脚杯深蹲", "腿", ["goblet squat"]),

    # ── 肩 ──
    ("推举", "肩", ["overhead press", "shoulder press", "杠铃推举", "ohp"]),
    ("哑铃推举", "肩", ["dumbbell press", "dumbbell shoulder press", "db press"]),
    ("侧平举", "肩", ["lateral raise", "哑铃侧平举"]),
    ("前平举", "肩", ["front raise", "哑铃前平举"]),
    ("俯身飞鸟", "肩", ["rear delt fly", "反向飞鸟", "reverse fly"]),
    ("面拉", "肩", ["face pull"]),
    ("阿诺德推举", "肩", ["arnold press"]),
    ("直立划船", "肩", ["upright row"]),
    ("杠铃耸肩", "肩", ["shrug", "barbell shrug"]),

    # ── 手臂 ──
    ("二头弯举", "手臂", ["bicep curl", "curl", "哑铃弯举", "杠铃弯举", "barbell curl"]),
    ("锤式弯举", "手臂", ["hammer curl"]),
    ("集中弯举", "手臂", ["concentration curl"]),
    ("牧师凳弯举", "手臂", ["preacher curl"]),
    ("三头下压", "手臂", ["tricep pushdown", "cable pushdown"]),
    ("绳索下压", "手臂", ["rope pushdown", "tricep rope"]),
    ("窄距卧推", "手臂", ["close grip bench press", "窄握卧推"]),
    ("法式弯举", "手臂", ["skull crusher", "仰卧臂屈伸"]),

    # ── 核心 ──
    ("卷腹", "核心", ["crunch", "ab crunch"]),
    ("平板支撑", "核心", ["plank"]),
    ("仰卧起坐", "核心", ["sit up", "situp"]),
    ("悬垂举腿", "核心", ["hanging leg raise"]),
    ("俄罗斯转体", "核心", ["russian twist"]),
    ("两头起", "核心", ["v-up", "v up"]),

    # ── 有氧 ──
    ("跑步", "有氧", ["run", "running", "jogging", "慢跑", "treadmill", "跑步机"]),
    ("跳绳", "有氧", ["jump rope", "skipping"]),
    ("骑行", "有氧", ["cycling", "bike", "biking", "动感单车", "spinning"]),
    ("椭圆机", "有氧", ["elliptical", "elliptical machine"]),
    ("划船机", "有氧", ["rowing machine", "rower", "rowing"]),
    ("游泳", "有氧", ["swim", "swimming"]),
]


async def seed_exercise_catalog() -> None:
    """应用首次启动时向 exercise_catalog 表填充预定义动作条目。

    幂等设计：插入前先查询该动作名是否已存在，已存在则跳过。
    因此无论重启多少次，都不会产生重复数据。

    执行时机：在 init_db() 建表之后调用。新条目会被批量提交到一个事务中。
    """
    async with async_session_factory() as session:
        inserted = 0
        for name, category, aliases in _EXERCISES:
            # 查询是否已存在同名动作，避免重复插入
            existing = await session.execute(
                select(ExerciseCatalog.id).where(ExerciseCatalog.name == name)
            )
            if existing.scalars().first():
                continue  # 已存在，跳过

            session.add(ExerciseCatalog(name=name, category=category, alias=aliases))
            inserted += 1

        if inserted:
            await session.commit()
            logger.info(f"Seeded exercise catalog: {inserted} new entries")
        else:
            logger.info("Exercise catalog already seeded — skipped")
