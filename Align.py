import contextlib
import io
import os
import time
from datetime import datetime, date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

LOGO_PATH = Path(__file__).parent / "assets" / "align_logo.png"
PAGE_ICON = str(LOGO_PATH) if LOGO_PATH.exists() else None

st.set_page_config(
    page_title="ALIGN | Workforce Capacity & Talent Matching",
    page_icon=PAGE_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT_GREEN = "#b7d591"


def render_live_clock():
    components.html(f"""
        <div id="align-clock" style="
            color:{ACCENT_GREEN}; font-weight:600; font-family:sans-serif;
            font-size:0.85rem; padding:2px 0;">
        </div>
        <script>
            function updateAlignClock() {{
                const el = document.getElementById("align-clock");
                if (el) {{ el.innerText = new Date().toLocaleString(); }}
            }}
            updateAlignClock();
            setInterval(updateAlignClock, 1000);
        </script>
    """, height=28)


STANDARD_WEEKLY_CAPACITY = 40

DATA_FILE_PATH = Path(__file__).parent / "align_data.xlsx"


def load_status(utilization: float) -> str:
    optimal_threshold = st.session_state.get("threshold_optimal", 80)
    burnout_threshold = st.session_state.get("threshold_burnout", 100)
    if utilization > burnout_threshold:
        return "Burnout Risk"
    return "Optimal Load" if utilization >= optimal_threshold else "Available"


STATUS_COLOR = {"Burnout Risk": "#E63946", "Optimal Load": "#F4A300", "Available": "#2A9D8F"}


def format_date_cell(val) -> str:
    if val is None or val == "":
        return ""
    if hasattr(val, "strftime"):
        return val.strftime("%Y-%m-%d")
    return str(val).strip()


def parse_skills(skills_str: str) -> set:
    return {s.strip() for s in str(skills_str).split(",") if s.strip()} if skills_str else set()


def rank_candidates_by_skills(pool: pd.DataFrame, required_skills: list) -> pd.DataFrame:
    if pool.empty:
        return pool
    required_set = set(required_skills)
    ranked = pool.copy()
    ranked["Skill Match"] = ranked["Skills"].apply(lambda s: len(parse_skills(s) & required_set))
    return ranked.sort_values(["Skill Match", "Utilization %"], ascending=[False, True])


def _ensure_float_hours(roster: pd.DataFrame) -> None:
    for col in ["Core Project Hours", "Ticket Hours", "Ad-hoc Hours", "Total Logged Hours"]:
        if col in roster.columns and roster[col].dtype != "float64":
            roster[col] = roster[col].astype(float)


def add_hours(emp_id: str, hrs: float) -> None:
    r = st.session_state.roster
    _ensure_float_hours(r)
    idx = r["Employee ID"] == emp_id
    r.loc[idx, "Core Project Hours"] += hrs
    r.loc[idx, "Total Logged Hours"] = r.loc[idx, "Core Project Hours"] + r.loc[idx, "Ticket Hours"] + r.loc[idx, "Ad-hoc Hours"]
    r.loc[idx, "Utilization %"] = round((r.loc[idx, "Total Logged Hours"] / STANDARD_WEEKLY_CAPACITY) * 100, 1)


def add_hours_by_name(name: str, hrs: float) -> None:
    r = st.session_state.roster
    _ensure_float_hours(r)
    idx = r["Name"].str.strip().str.lower() == name.strip().lower()
    if not idx.any():
        return
    r.loc[idx, "Core Project Hours"] = (r.loc[idx, "Core Project Hours"] + hrs).clip(lower=0)
    r.loc[idx, "Total Logged Hours"] = r.loc[idx, "Core Project Hours"] + r.loc[idx, "Ticket Hours"] + r.loc[idx, "Ad-hoc Hours"]
    r.loc[idx, "Utilization %"] = round((r.loc[idx, "Total Logged Hours"] / STANDARD_WEEKLY_CAPACITY) * 100, 1)


def mark_task_complete(record: dict) -> str:
    names = [n.strip() for n in str(record.get("assigned_to", "")).split(",") if n.strip()]
    hours_str = str(record.get("hours", "")).strip()
    hours_list = []
    if hours_str:
        try:
            hours_list = [float(h.strip()) for h in hours_str.split(",") if h.strip()]
        except ValueError:
            hours_list = []
    if len(hours_list) != len(names):
        hours_list = [0.0] * len(names)

    for name, hrs in zip(names, hours_list):
        if hrs:
            add_hours_by_name(name, -hrs)
        idx = st.session_state.roster["Name"].str.strip().str.lower() == name.lower()
        st.session_state.roster.loc[idx, "Completed Milestones Count"] += 1

    record["completed"] = True
    freed = sum(hours_list)
    return f"{freed:g} hrs freed from {', '.join(names)}" if freed else f"milestone credited to {', '.join(names)}"


NEAR_FULL_THRESHOLD = 90


def find_person_least_critical_task(person_name: str, exclude_record: dict = None):
    name_lc = person_name.strip().lower()

    def is_assigned(assigned_to: str) -> bool:
        return name_lc in {n.strip().lower() for n in str(assigned_to).split(",")}

    candidates = []
    for p in st.session_state.projects:
        for s in p["subtasks"]:
            if s is not exclude_record and not s.get("completed") and is_assigned(s.get("assigned_to", "")):
                candidates.append(("subtask", p, s))
    for t in st.session_state.operational_tasks:
        if t is not exclude_record and not t.get("completed") and is_assigned(t.get("assigned_to", "")):
            candidates.append(("ops", None, t))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[2].get("criticality") if isinstance(c[2].get("criticality"), int) else 0)
    return candidates[0]


def attempt_task_reassignment(person_name: str, exclude_record: dict = None) -> str:
    found = find_person_least_critical_task(person_name, exclude_record)
    if found is None:
        return ""
    kind, project_ref, task_record = found

    if kind == "subtask":
        domain = project_ref["domain"]
    else:
        person_rows = st.session_state.roster[st.session_state.roster["Name"].str.strip().str.lower() == person_name.strip().lower()]
        if person_rows.empty:
            return ""
        domain = person_rows.iloc[0]["Primary Domain"]

    domain_pool = st.session_state.roster[
        (st.session_state.roster["Primary Domain"] == domain)
        & (st.session_state.roster["Name"].str.strip().str.lower() != person_name.strip().lower())
    ]
    if domain_pool.empty:
        return ""

    names = [n.strip() for n in str(task_record.get("assigned_to", "")).split(",") if n.strip()]
    hours_list_raw = str(task_record.get("hours", "")).split(",")
    try:
        hours_list = [float(h.strip()) for h in hours_list_raw if h.strip()]
    except ValueError:
        hours_list = []
    if len(hours_list) != len(names):
        hours_list = [0.0] * len(names)
    try:
        person_slot = [n.lower() for n in names].index(person_name.strip().lower())
        hrs_for_person = hours_list[person_slot]
    except ValueError:
        return ""

    required_skills = parse_skills(task_record.get("skills_needed", ""))
    ranked = rank_candidates_by_skills(domain_pool, list(required_skills))
    safe_ranked = ranked[ranked["Utilization %"] + (hrs_for_person / STANDARD_WEEKLY_CAPACITY * 100) <= 100]
    if safe_ranked.empty:
        return ""

    new_person = safe_ranked.iloc[0]["Name"]
    add_hours_by_name(person_name, -hrs_for_person)
    add_hours_by_name(new_person, hrs_for_person)

    updated_names = [new_person if n.lower() == person_name.strip().lower() else n for n in names]
    task_record["assigned_to"] = ", ".join(updated_names)

    task_label = task_record.get("subtask") or task_record.get("task")
    return f"Reassigned '{task_label}' from {person_name} to {new_person} to make room for a higher-priority task"


def backfill_missing_dates(default_days_out: int = 7) -> int:
    today_str = str(date.today())
    due_str = str(date.today() + timedelta(days=default_days_out))
    count = 0
    for p in st.session_state.projects:
        for s in p["subtasks"]:
            if not s.get("assigned_date") or not s.get("due_date"):
                s["assigned_date"] = s.get("assigned_date") or today_str
                s["due_date"] = s.get("due_date") or due_str
                count += 1
    for t in st.session_state.operational_tasks:
        if not t.get("assigned_date") or not t.get("due_date"):
            t["assigned_date"] = t.get("assigned_date") or today_str
            t["due_date"] = t.get("due_date") or due_str
            count += 1
    return count


def build_task_timeline_df(projects: list, operational_tasks: list) -> pd.DataFrame:
    rows = []

    def add_rows(name_key, items, project_label_fn):
        for item in items:
            start = item.get("assigned_date", "")
            due = item.get("due_date", "")
            if not start or not due:
                continue
            names = [n.strip() for n in str(item.get("assigned_to", "")).split(",") if n.strip()]
            for person in names:
                rows.append({
                    "Employee": person, "Task": item[name_key], "Project": project_label_fn(item),
                    "Start": start, "Finish": due, "Completed": item.get("completed", False),
                })

    for p in projects:
        add_rows("subtask", p["subtasks"], lambda s, proj=p["project"]: proj)
    add_rows("task", operational_tasks, lambda t: "(Operational)")

    if not rows:
        return pd.DataFrame(columns=["Employee", "Task", "Project", "Start", "Finish", "Completed", "Status"])

    tdf = pd.DataFrame(rows)
    tdf["Start"] = pd.to_datetime(tdf["Start"], errors="coerce")
    tdf["Finish"] = pd.to_datetime(tdf["Finish"], errors="coerce")
    tdf = tdf.dropna(subset=["Start", "Finish"])
    tdf["Status"] = tdf["Completed"].map({True: "Completed", False: "In Progress"})
    return tdf


def compute_recognition_spotlight(roster: pd.DataFrame) -> dict:
    healthy = roster[roster["Status"] != "Burnout Risk"].sort_values(
        "Completed Milestones Count", ascending=False
    )
    return None if healthy.empty else healthy.iloc[0].to_dict()


def render_task_row(record: dict, name_key: str, row_key: str, show_assignee: bool = True, can_confirm: bool = False):
    is_done = record.get("completed", False)
    is_pending = record.get("pending_completion", False) and not is_done
    cols = st.columns([3, 2, 1.3]) if show_assignee else st.columns([4, 1.3])
    col_info = cols[0]
    col_assignee = cols[1] if show_assignee else None
    col_action = cols[-1]
    with col_info:
        label = record[name_key]
        st.markdown(f"~~{label}~~" if is_done else label)
        criticality = record.get("criticality", "")
        if criticality:
            crit_color = {5: "#E63946", 4: "#E63946", 3: "#F4A300"}.get(int(criticality), "#2A9D8F")
            st.markdown(
                f"<span style='color:{crit_color}; font-size:0.85em; font-weight:600;'>Criticality {criticality}/5</span>",
                unsafe_allow_html=True,
            )
        due_date_str = record.get("due_date", "")
        if due_date_str:
            try:
                due_dt = datetime.strptime(due_date_str, "%Y-%m-%d").date()
                is_overdue = (not is_done) and due_dt < date.today()
                due_label = f"Due {due_date_str}" + (" \u2014 OVERDUE" if is_overdue else "")
                due_color = "#E63946" if is_overdue else "#666666"
                st.markdown(
                    f"<span style='color:{due_color}; font-size:0.85em;'>{due_label}</span>",
                    unsafe_allow_html=True,
                )
            except ValueError:
                pass
        skills_needed = record.get("skills_needed", "")
        if skills_needed:
            st.caption(f"Skills needed: {skills_needed}")
        if is_pending:
            st.markdown(
                "<span style='color:#F4A300; font-size:0.85em; font-weight:600;'>Pending confirmation</span>",
                unsafe_allow_html=True,
            )
            note = record.get("completion_note", "")
            if note:
                st.caption(f"Note: {note}")
    if show_assignee:
        with col_assignee:
            st.write(record.get("assigned_to", ""))
    with col_action:
        if is_done:
            st.caption("Completed")
        elif is_pending:
            if can_confirm:
                if st.button("Confirm Completion", key=f"{row_key}_confirm", use_container_width=True, type="primary"):
                    summary = mark_task_complete(record)
                    record["pending_completion"] = False
                    st.session_state.assignment_log.append(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] "
                        f"Confirmed complete '{label}': {summary}"
                    )
                    st.session_state.setdefault("structured_log", []).append({
                        "timestamp": datetime.now(), "action": "Confirmed Complete", "task_name": label,
                        "domain": "", "hours": "", "assignees": record.get("assigned_to", ""),
                        "criticality": record.get("criticality", ""),
                    })
                    try:
                        save_data_to_workbook(DATA_FILE_PATH)
                        st.session_state.data_file_mtime = DATA_FILE_PATH.stat().st_mtime
                    except Exception as exc:
                        st.warning(f"Confirmed in-app, but couldn't sync to file: {exc}")
                    st.rerun()
            else:
                st.caption("Awaiting confirmation")
        else:
            with st.expander("Mark Complete"):
                note_text = st.text_area("What did you do?", key=f"{row_key}_note", label_visibility="visible")
                if st.button("Submit for Confirmation", key=f"{row_key}_submit", use_container_width=True, type="primary"):
                    record["pending_completion"] = True
                    record["completion_note"] = note_text.strip()
                    st.session_state.assignment_log.append(
                        f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] "
                        f"Submitted '{label}' for completion confirmation"
                    )
                    st.session_state.setdefault("structured_log", []).append({
                        "timestamp": datetime.now(), "action": "Submitted for Confirmation", "task_name": label,
                        "domain": "", "hours": "", "assignees": record.get("assigned_to", ""),
                        "criticality": record.get("criticality", ""),
                    })
                    try:
                        save_data_to_workbook(DATA_FILE_PATH)
                        st.session_state.data_file_mtime = DATA_FILE_PATH.stat().st_mtime
                    except Exception as exc:
                        st.warning(f"Saved in-app, but couldn't sync to file: {exc}")
                    st.rerun()


def person_tasks(name: str) -> tuple:
    name_lc = name.strip().lower()

    def is_assigned(assigned_to: str) -> bool:
        return name_lc in {n.strip().lower() for n in str(assigned_to).split(",")}

    my_subtasks = [
        (p["project"], s)
        for p in st.session_state.projects for s in p["subtasks"]
        if is_assigned(s["assigned_to"])
    ]
    my_ops = [
        t for t in st.session_state.operational_tasks
        if is_assigned(t["assigned_to"])
    ]
    return my_subtasks, my_ops


def load_workbook_data(uploaded_file) -> dict:
    sheets = pd.read_excel(uploaded_file, sheet_name=None, engine="openpyxl")

    required_sheets = ["Employees", "Skill Catalog", "Projects", "Operational Tasks"]
    missing = [s for s in required_sheets if s not in sheets]
    if missing:
        raise ValueError(f"Missing sheet(s): {', '.join(missing)}")

    emp_df = sheets["Employees"].fillna("")
    skill_df = sheets["Skill Catalog"].fillna("")
    proj_df = sheets["Projects"].fillna("")
    ops_df = sheets["Operational Tasks"].fillna("")

    roster = emp_df.copy()
    for col in ["Core Project Hours", "Ticket Hours", "Ad-hoc Hours"]:
        roster[col] = pd.to_numeric(roster[col], errors="coerce").fillna(0.0).astype(float)
    roster["Completed Milestones Count"] = pd.to_numeric(
        roster["Completed Milestones Count"], errors="coerce"
    ).fillna(0).astype(int)
    roster["Weekly Capacity"] = STANDARD_WEEKLY_CAPACITY
    roster["Skills"] = roster["Skills"].astype(str).replace("nan", "")
    roster["Total Logged Hours"] = roster["Core Project Hours"] + roster["Ticket Hours"] + roster["Ad-hoc Hours"]
    roster["Utilization %"] = round((roster["Total Logged Hours"] / STANDARD_WEEKLY_CAPACITY) * 100, 1)

    domains = list(dict.fromkeys(d for d in skill_df["Domain"].tolist() if d))
    if not domains:
        raise ValueError("The 'Skill Catalog' sheet has no rows. Add at least one Domain/Skill row.")
    if roster.empty:
        raise ValueError("The 'Employees' sheet has no rows. Add at least one employee.")
    skills_by_domain = {d: skill_df.loc[skill_df["Domain"] == d, "Skill"].tolist() for d in domains}

    roles_by_domain = {}
    for d in domains:
        roles = roster.loc[roster["Primary Domain"] == d, "Role"].dropna().unique().tolist()
        roles_by_domain[d] = [r for r in roles if r] or ["Team Member"]

    projects = []
    if not proj_df.empty:
        for (proj_name, domain), group in proj_df.groupby(["Project", "Domain"], sort=False):
            subtasks = [
                {
                    "subtask": r["Subtask"], "assigned_to": r["Assigned To"],
                    "skills_needed": r.get("Skills Needed", ""),
                    "hours": str(r.get("Hours", "")), "completed": bool(r.get("Completed", False)),
                    "criticality": r.get("Criticality", ""),
                    "assigned_date": format_date_cell(r.get("Start Date", "")),
                    "due_date": format_date_cell(r.get("Due Date", "")),
                    "pending_completion": bool(r.get("Pending Completion", False)),
                    "completion_note": r.get("Completion Note", ""),
                }
                for _, r in group.iterrows()
            ]
            projects.append({"domain": domain, "project": proj_name, "subtasks": subtasks})

    operational_tasks = [
        {
            "task": r["Task"], "assigned_to": r["Assigned To"],
            "skills_needed": r.get("Skills Needed", ""),
            "hours": str(r.get("Hours", "")), "completed": bool(r.get("Completed", False)),
            "criticality": r.get("Criticality", ""),
            "assigned_date": format_date_cell(r.get("Start Date", "")),
            "due_date": format_date_cell(r.get("Due Date", "")),
            "pending_completion": bool(r.get("Pending Completion", False)),
            "completion_note": r.get("Completion Note", ""),
        }
        for _, r in ops_df.iterrows()
    ]

    return {
        "roster": roster,
        "domains": domains,
        "skills_by_domain": skills_by_domain,
        "roles_by_domain": roles_by_domain,
        "projects": projects,
        "operational_tasks": operational_tasks,
    }


@contextlib.contextmanager
def _file_lock(lock_path: Path, timeout: float = 5.0, stale_after: float = 10.0):
    deadline = time.time() + timeout
    acquired = False
    while time.time() < deadline:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            acquired = True
            break
        except FileExistsError:
            try:
                age = time.time() - lock_path.stat().st_mtime
            except FileNotFoundError:
                continue
            if age > stale_after:
                lock_path.unlink(missing_ok=True)
                continue
            time.sleep(0.2)

    if not acquired:
        yield False
        return
    try:
        yield True
    finally:
        lock_path.unlink(missing_ok=True)


def save_data_to_workbook(path: Path) -> None:
    roster_out = st.session_state.roster[[
        "Employee ID", "Name", "Role", "Primary Domain", "Experience Level",
        "Skills", "Weekly Capacity", "Core Project Hours", "Ticket Hours",
        "Ad-hoc Hours", "Completed Milestones Count",
    ]]

    skill_rows = [
        {"Domain": d, "Skill": s}
        for d, skills in st.session_state.skills_by_domain.items()
        for s in skills
    ]
    skill_out = pd.DataFrame(skill_rows, columns=["Domain", "Skill"])

    project_rows = [
        {"Project": p["project"], "Domain": p["domain"], "Subtask": s["subtask"],
         "Assigned To": s["assigned_to"], "Skills Needed": s.get("skills_needed", ""),
         "Hours": s.get("hours", ""), "Completed": s.get("completed", False),
         "Criticality": s.get("criticality", ""),
         "Start Date": s.get("assigned_date", ""), "Due Date": s.get("due_date", ""),
         "Pending Completion": s.get("pending_completion", False),
         "Completion Note": s.get("completion_note", "")}
        for p in st.session_state.projects for s in p["subtasks"]
    ]
    project_out = pd.DataFrame(project_rows, columns=["Project", "Domain", "Subtask", "Assigned To", "Skills Needed", "Hours", "Completed", "Criticality", "Start Date", "Due Date", "Pending Completion", "Completion Note"])

    ops_rows = [
        {"Task": t["task"], "Assigned To": t["assigned_to"], "Skills Needed": t.get("skills_needed", ""),
         "Hours": t.get("hours", ""), "Completed": t.get("completed", False),
         "Criticality": t.get("criticality", ""),
         "Start Date": t.get("assigned_date", ""), "Due Date": t.get("due_date", ""),
         "Pending Completion": t.get("pending_completion", False),
         "Completion Note": t.get("completion_note", "")}
        for t in st.session_state.operational_tasks
    ]
    ops_out = pd.DataFrame(ops_rows, columns=["Task", "Assigned To", "Skills Needed", "Hours", "Completed", "Criticality", "Start Date", "Due Date", "Pending Completion", "Completion Note"])

    tmp_path = path.with_suffix(".tmp.xlsx")
    lock_path = path.with_suffix(".lock")
    with _file_lock(lock_path):
        with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
            roster_out.to_excel(writer, sheet_name="Employees", index=False)
            skill_out.to_excel(writer, sheet_name="Skill Catalog", index=False)
            project_out.to_excel(writer, sheet_name="Projects", index=False)
            ops_out.to_excel(writer, sheet_name="Operational Tasks", index=False)
        tmp_path.replace(path)


def build_powerbi_export(
    roster: pd.DataFrame, projects: list, operational_tasks: list, structured_log: list,
) -> io.BytesIO:
    employees_out = roster.copy()
    employees_out["Status"] = employees_out["Utilization %"].apply(load_status)

    task_rows = []
    for p in projects:
        for s in p["subtasks"]:
            task_rows.append({
                "Type": "Core Project", "Project": p["project"], "Domain": p["domain"],
                "Task": s["subtask"], "Assigned To": s.get("assigned_to", ""),
                "Skills Needed": s.get("skills_needed", ""), "Hours": s.get("hours", ""),
                "Completed": s.get("completed", False), "Criticality": s.get("criticality", ""),
                "Start Date": s.get("assigned_date", ""), "Due Date": s.get("due_date", ""),
                "Pending Completion": s.get("pending_completion", False),
                "Completion Note": s.get("completion_note", ""),
            })
    for t in operational_tasks:
        task_rows.append({
            "Type": "Operational", "Project": "", "Domain": "",
            "Task": t["task"], "Assigned To": t.get("assigned_to", ""),
            "Skills Needed": t.get("skills_needed", ""), "Hours": t.get("hours", ""),
            "Completed": t.get("completed", False), "Criticality": t.get("criticality", ""),
            "Start Date": t.get("assigned_date", ""), "Due Date": t.get("due_date", ""),
            "Pending Completion": t.get("pending_completion", False),
            "Completion Note": t.get("completion_note", ""),
        })
    tasks_out = pd.DataFrame(task_rows, columns=[
        "Type", "Project", "Domain", "Task", "Assigned To", "Skills Needed", "Hours", "Completed",
        "Criticality", "Start Date", "Due Date", "Pending Completion", "Completion Note",
    ])

    log_cols = ["timestamp", "action", "task_name", "domain", "hours", "assignees", "criticality"]
    log_out = pd.DataFrame(structured_log, columns=log_cols) if structured_log else pd.DataFrame(columns=log_cols)
    log_out = log_out.rename(columns={
        "timestamp": "Timestamp", "action": "Action", "task_name": "Task Name",
        "domain": "Domain", "hours": "Hours", "assignees": "Assignees", "criticality": "Criticality",
    })

    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        employees_out.to_excel(writer, sheet_name="Employees", index=False)
        tasks_out.to_excel(writer, sheet_name="Tasks", index=False)
        log_out.to_excel(writer, sheet_name="Activity Log", index=False)
    buffer.seek(0)
    return buffer


REQUIRED_DATA_KEYS = ["roster", "domains", "skills_by_domain", "roles_by_domain", "projects", "operational_tasks"]


def show_upload_screen():
    st.title("ALIGN: Load Department Data")
    st.write(
        "Upload your filled-in data workbook to begin. It will be saved locally and "
        "auto-loaded every time you reopen the app, no need to upload it again."
    )
    uploaded = st.file_uploader("Upload data workbook (.xlsx)", type=["xlsx"])
    if uploaded is not None:
        try:
            data = load_workbook_data(uploaded)
        except Exception as exc:
            st.error(f"Could not read this file: {exc}")
            return
        for key in REQUIRED_DATA_KEYS:
            st.session_state[key] = data[key]
        st.session_state.assignment_log = []
        try:
            save_data_to_workbook(DATA_FILE_PATH)
            st.session_state.data_file_mtime = DATA_FILE_PATH.stat().st_mtime
        except Exception as exc:
            st.warning(f"Loaded, but couldn't save a local copy for next time: {exc}")
        st.rerun()


values = {key: st.session_state.get(key) for key in REQUIRED_DATA_KEYS}

if any(v is None for v in values.values()) and DATA_FILE_PATH.exists():
    try:
        data = load_workbook_data(str(DATA_FILE_PATH))
        for key in REQUIRED_DATA_KEYS:
            st.session_state[key] = data[key]
        st.session_state.setdefault("assignment_log", [])
        st.session_state.data_file_mtime = DATA_FILE_PATH.stat().st_mtime
        values = {key: st.session_state.get(key) for key in REQUIRED_DATA_KEYS}
    except Exception:
        pass

if any(v is None for v in values.values()):
    for key in REQUIRED_DATA_KEYS:
        st.session_state.pop(key, None)
    show_upload_screen()
    st.stop()

if DATA_FILE_PATH.exists():
    try:
        current_mtime = DATA_FILE_PATH.stat().st_mtime
        if current_mtime != st.session_state.get("data_file_mtime"):
            data = load_workbook_data(str(DATA_FILE_PATH))
            for key in REQUIRED_DATA_KEYS:
                st.session_state[key] = data[key]
            st.session_state.data_file_mtime = current_mtime
            values = {key: st.session_state.get(key) for key in REQUIRED_DATA_KEYS}
    except Exception:
        pass

DOMAINS = values["domains"]
SKILLS_BY_DOMAIN = values["skills_by_domain"]
ROLES_BY_DOMAIN = values["roles_by_domain"]
df = values["roster"]

if "assignment_log" not in st.session_state:
    st.session_state.assignment_log = []
if "user_role" not in st.session_state:
    st.session_state.user_role = None

try:
    df["Status"] = df["Utilization %"].apply(load_status)
except Exception as exc:
    for key in REQUIRED_DATA_KEYS:
        st.session_state.pop(key, None)
    st.error(f"The loaded data looks invalid ({exc}). Please re-upload the workbook.")
    show_upload_screen()
    st.stop()


def render_role_selection():
    render_live_clock()
    _, center_col, _ = st.columns([1, 2, 1])
    with center_col:
        if LOGO_PATH.exists():
            _, logo_mid, _ = st.columns([1, 3, 1])
            with logo_mid:
                st.image(str(LOGO_PATH), use_container_width=True)
        st.markdown("<h1 style='text-align: center;'>Welcome to ALIGN</h1>", unsafe_allow_html=True)
        st.markdown(
            "<p style='text-align: center; color: gray;'>"
            "Select how you'd like to use the platform to continue.</p>",
            unsafe_allow_html=True,
        )

    st.divider()
    col_aligner, col_achiever = st.columns(2)

    with col_aligner:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>Aligner</h3>", unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align: center; color: gray;'>Chief / Manager view</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='text-align: center;'>Full capacity visibility across the team: "
                "burnout heatmap, skill-based task matching, and recognition tracking.</p>",
                unsafe_allow_html=True,
            )
            if st.button("Continue as Aligner", use_container_width=True, type="primary"):
                st.session_state.user_role = "aligner"
                st.rerun()

    with col_achiever:
        with st.container(border=True):
            st.markdown("<h3 style='text-align: center;'>Achiever</h3>", unsafe_allow_html=True)
            st.markdown(
                "<p style='text-align: center; color: gray;'>Team Member view</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<p style='text-align: center;'>Your personal workload, assigned tasks, "
                "and milestones.</p>",
                unsafe_allow_html=True,
            )
            if st.button("Continue as Achiever", use_container_width=True, type="primary"):
                st.session_state.user_role = "achiever"
                st.rerun()


if st.session_state.user_role is None:
    render_role_selection()
    st.stop()

with st.sidebar:
    if LOGO_PATH.exists():
        st.image(str(LOGO_PATH), width=90)
    st.caption("ALIGN: Workforce Capacity & Talent Matching")
    render_live_clock()
    st.markdown("---")

    current_role = st.session_state.get("user_role") or "Guest"
    st.caption(f"Signed in as: **{current_role.title()}**")
    if st.button("Switch Role", use_container_width=True, type="primary"):
        st.session_state.user_role = None
        st.rerun()
    st.markdown("---")

    st.caption(f"Data file: `{DATA_FILE_PATH.name}`")
    if st.button("Reload from file", use_container_width=True):
        try:
            data = load_workbook_data(str(DATA_FILE_PATH))
        except Exception as exc:
            st.error(f"Couldn't reload: {exc}")
            data = None
        if data is not None:
            for key in REQUIRED_DATA_KEYS:
                st.session_state[key] = data[key]
            st.session_state.data_file_mtime = DATA_FILE_PATH.stat().st_mtime
            st.rerun()

    auto_refresh = st.checkbox(
        "Auto-refresh every 30s",
        value=st.session_state.get("auto_refresh_enabled", False),
        help="Automatically reloads the page periodically to pick up changes "
             "saved by others without clicking anything. Turn this off while "
             "filling out a form, since a refresh discards unsaved typing.",
    )
    st.session_state.auto_refresh_enabled = auto_refresh
    if auto_refresh:
        components.html(
            "<script>setTimeout(function(){ window.parent.location.reload(); }, 30000);</script>",
            height=0,
        )
    st.markdown("---")


if st.session_state.user_role == "achiever":
    st.title("Achiever: Team Member View")
    st.divider()

    if not st.session_state.get("achiever_name"):
        st.caption("Enter your name to continue.")
        with st.form("achiever_name_gate"):
            name_input = st.text_input("Your Name", placeholder="e.g. Sara")
            continue_clicked = st.form_submit_button("Continue", type="primary")
        if continue_clicked:
            clean_name = name_input.strip()
            if not clean_name:
                st.warning("Please enter your name to continue.")
            else:
                st.session_state.achiever_name = clean_name
                st.rerun()
        st.stop()

    my_name = st.session_state.achiever_name
    roster = st.session_state.roster
    me_mask = roster["Name"].str.strip().str.lower() == my_name.lower()

    name_col, switch_col = st.columns([4, 1])
    with name_col:
        st.subheader(f"Welcome, {my_name}")
    with switch_col:
        if st.button("Not you?", use_container_width=True):
            st.session_state.achiever_name = None
            st.rerun()

    if not me_mask.any():
        st.info("We don't have a profile for you yet. Set one up to continue.")

        default_domain = st.session_state.get("achiever_domain", DOMAINS[0])
        domain_index = DOMAINS.index(default_domain) if default_domain in DOMAINS else 0
        domain_input = st.selectbox("Your Domain", DOMAINS, index=domain_index)

        with st.form("achiever_registration"):
            level_options = ["PDP", "Senior"]
            level_input = st.selectbox("Experience Level", level_options)
            skills_input = st.multiselect(
                "Your Skills", options=SKILLS_BY_DOMAIN[domain_input],
                help="Scoped to your selected domain.",
            )
            registered = st.form_submit_button("Save My Profile", type="primary")

        if registered:
            st.session_state.achiever_domain = domain_input
            st.session_state.achiever_level = level_input
            st.session_state.achiever_skills = skills_input
            rng = np.random.default_rng()
            new_row = {
                "Employee ID": f"EMP-{2000 + len(roster)}",
                "Name": my_name,
                "Role": rng.choice(ROLES_BY_DOMAIN[domain_input]),
                "Primary Domain": domain_input,
                "Experience Level": level_input,
                "Skills": ", ".join(skills_input),
                "Weekly Capacity": STANDARD_WEEKLY_CAPACITY,
                "Core Project Hours": 0.0,
                "Ticket Hours": 0.0,
                "Ad-hoc Hours": 0.0,
                "Total Logged Hours": 0.0,
                "Completed Milestones Count": 0,
                "Utilization %": 0.0,
            }
            st.session_state.roster = pd.concat([roster, pd.DataFrame([new_row])], ignore_index=True)
            st.success(f"Welcome, {my_name}! You've been added under **{domain_input}** ({level_input}).")
            try:
                save_data_to_workbook(DATA_FILE_PATH)
                st.session_state.data_file_mtime = DATA_FILE_PATH.stat().st_mtime
            except Exception as exc:
                st.warning(f"Saved in-app, but couldn't sync to file: {exc}")
            st.rerun()
        st.stop()

    me = roster.loc[me_mask].iloc[0]

    with st.expander("Update my Domain / Experience Level / Skills"):
        edit_domain_index = DOMAINS.index(me["Primary Domain"]) if me["Primary Domain"] in DOMAINS else 0
        edit_domain = st.selectbox("Your Domain", DOMAINS, index=edit_domain_index, key="edit_domain")

        with st.form("achiever_edit_form"):
            level_options = ["PDP", "Senior"]
            edit_level_index = level_options.index(me["Experience Level"]) if me["Experience Level"] in level_options else 0
            edit_level = st.selectbox("Experience Level", level_options, index=edit_level_index)

            current_skills = [s for s in parse_skills(me.get("Skills", "")) if s in SKILLS_BY_DOMAIN[edit_domain]]
            edit_skills = st.multiselect("Your Skills", options=SKILLS_BY_DOMAIN[edit_domain], default=current_skills)

            save_edits = st.form_submit_button("Save Changes", type="primary")

        if save_edits:
            roster.loc[me_mask, "Primary Domain"] = edit_domain
            roster.loc[me_mask, "Experience Level"] = edit_level
            roster.loc[me_mask, "Skills"] = ", ".join(edit_skills)
            st.success(f"Updated: now classified under **{edit_domain}** ({edit_level}).")
            try:
                save_data_to_workbook(DATA_FILE_PATH)
                st.session_state.data_file_mtime = DATA_FILE_PATH.stat().st_mtime
            except Exception as exc:
                st.warning(f"Saved in-app, but couldn't sync to file: {exc}")
            st.rerun()

    me_status = load_status(me["Utilization %"])

    st.divider()
    st.subheader(f"{me['Name']}'s Capacity Snapshot")

    snap1, snap2, snap3 = st.columns(3)
    snap1.metric("Domain", me["Primary Domain"])
    snap2.metric("Utilization", f"{me['Utilization %']}%")
    snap3.metric("Status", me_status)
    st.caption(f"**Level:** {me['Experience Level']}  |  **Skills:** {me.get('Skills') or 'None'}")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[me["Name"]], y=[me["Utilization %"]], marker_color=STATUS_COLOR[me_status],
        text=[f"{me['Utilization %']}%"], textposition="outside", width=[0.4],
    ))
    fig.add_hline(y=100, line_dash="dash", line_color="#E63946",
                  annotation_text="Burnout Threshold (100%)", annotation_position="top left")
    fig.update_layout(yaxis_title="Utilization %", xaxis_title="", height=350,
                       margin=dict(t=30, b=20), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("My Assigned Work")
    st.caption("Mark a task complete once it's done. This frees up your hours and credits a milestone.")
    my_subtasks, my_ops = person_tasks(me["Name"])

    if not my_subtasks and not my_ops:
        st.caption("No projects or tasks assigned yet.")
    else:
        if my_subtasks:
            st.caption("Project subtasks:")
            for i, (project_name, subtask) in enumerate(my_subtasks):
                st.caption(f"Project: {project_name}")
                render_task_row(subtask, "subtask", f"achiever_subtask_{i}", show_assignee=False)
                st.markdown("---")
        if my_ops:
            st.caption("Operational tasks:")
            for i, task in enumerate(my_ops):
                render_task_row(task, "task", f"achiever_optask_{i}", show_assignee=False)
                st.markdown("---")

    st.stop()


tab_dashboard, tab_matching, tab_projects, tab_recognition, tab_raw = st.tabs([
    "Capacity & Burnout Dashboard",
    "Talent Matching Engine",
    "Projects & Operational Tasks",
    "Recognition & Milestone Hub",
    "Raw Roster Data",
])

with tab_dashboard:
    with st.expander("Adjust Risk Thresholds"):
        thresh_col1, thresh_col2 = st.columns(2)
        with thresh_col1:
            new_optimal = st.number_input(
                "Optimal Load starts at (%)",
                min_value=1, max_value=200,
                value=st.session_state.get("threshold_optimal", 80),
                help="Utilization at or above this is 'Optimal Load' instead of 'Available'.",
            )
        with thresh_col2:
            new_burnout = st.number_input(
                "Burnout Risk starts above (%)",
                min_value=1, max_value=300,
                value=st.session_state.get("threshold_burnout", 100),
                help="Utilization above this is 'Burnout Risk'.",
            )
        if new_optimal >= new_burnout:
            st.warning("Optimal threshold must be lower than the Burnout threshold. Not saved.")
        else:
            st.session_state.threshold_optimal = new_optimal
            st.session_state.threshold_burnout = new_burnout
            df["Status"] = df["Utilization %"].apply(load_status)

    total_workforce = len(df)
    overloaded = int((df["Utilization %"] > st.session_state.get("threshold_burnout", 100)).sum())
    available = int((df["Utilization %"] < st.session_state.get("threshold_optimal", 80)).sum())
    optimal = total_workforce - overloaded - available

    st.subheader("At a Glance")
    st.caption("The headline from every other tab, pulled into one place. Click a tab above for the full detail.")

    with st.container(border=True):
        st.markdown("**Summary**")
        total_subtasks = sum(len(p["subtasks"]) for p in st.session_state.projects)
        done_subtasks = sum(1 for p in st.session_state.projects for s in p["subtasks"] if s.get("completed"))
        total_ops = len(st.session_state.operational_tasks)
        done_ops = sum(1 for t in st.session_state.operational_tasks if t.get("completed"))
        star = compute_recognition_spotlight(df)
        spotlight_value = star["Name"] if star is not None else "\u2014"
        spotlight_delta = f"{star['Completed Milestones Count']} milestones" if star is not None else "No qualifying spotlight yet"

        row1_1, row1_2, row1_3, row1_4 = st.columns(4)
        row1_1.metric("Total Workforce", total_workforce)
        row1_2.metric(f"Overloaded (>{st.session_state.get('threshold_burnout', 100)}%)", overloaded)
        row1_3.metric("Optimal Load", optimal)
        row1_4.metric(f"Available (<{st.session_state.get('threshold_optimal', 80)}%)", available)

        row2_1, row2_2, row2_3, row2_4 = st.columns(4)
        row2_1.metric("Active Projects", len(st.session_state.projects))
        row2_2.metric("Project Subtasks Done", f"{done_subtasks} / {total_subtasks}")
        row2_3.metric("Operational Tasks Done", f"{done_ops} / {total_ops}")
        row2_4.metric("Spotlight", spotlight_value, spotlight_delta, delta_color="off")

    gauge_col, heatmap_col = st.columns([1, 2.5])

    with gauge_col:
        with st.container(border=True):
            avg_util = round(df["Utilization %"].mean(), 1) if total_workforce else 0.0
            gauge_util = go.Figure(go.Indicator(
                mode="gauge+number",
                value=avg_util,
                title={"text": "Average Utilization", "font": {"size": 16}},
                number={"suffix": "%"},
                gauge={
                    "axis": {"range": [0, 150]},
                    "bar": {"color": ACCENT_GREEN},
                    "steps": [
                        {"range": [0, 80], "color": "#2A9D8F33"},
                        {"range": [80, 100], "color": "#F4A30033"},
                        {"range": [100, 150], "color": "#E6394633"},
                    ],
                    "threshold": {"line": {"color": "#E63946", "width": 3}, "thickness": 0.8, "value": 100},
                },
            ))
            gauge_util.update_layout(height=260, margin=dict(t=50, b=10, l=25, r=25))
            st.plotly_chart(gauge_util, use_container_width=True)
            st.caption("Workforce-wide average, red line marks 100%.")

    with heatmap_col:
        with st.container(border=True):
            df_sorted = df.sort_values("Utilization %", ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_sorted["Name"],
                y=df_sorted["Utilization %"],
                marker_color=[STATUS_COLOR[s] for s in df_sorted["Status"]],
                text=df_sorted["Utilization %"].astype(str) + "%",
                textposition="outside",
                hovertext=(
                    "Domain: " + df_sorted["Primary Domain"] + "<br>"
                    + "Level: " + df_sorted["Experience Level"] + "<br>"
                    + "Logged Hrs: " + df_sorted["Total Logged Hours"].astype(str) + " / 40"
                ),
                hoverinfo="text+x+y",
                name="Utilization %",
            ))
            fig.add_hline(y=100, line_dash="dash", line_color="#E63946",
                          annotation_text="Burnout Threshold (100% / 40 hrs)", annotation_position="top left")
            fig.update_layout(
                title={"text": "Workload Ranking", "font": {"size": 16}},
                yaxis_title="Utilization %", xaxis_title="Engineer", height=260,
                margin=dict(t=50, b=10), showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                f"**Red** = Burnout Risk (>{st.session_state.get('threshold_burnout', 100)}%)   |   "
                f"**Yellow** = Optimal Load ({st.session_state.get('threshold_optimal', 80)}-{st.session_state.get('threshold_burnout', 100)}%)   |   "
                f"**Green** = Available (<{st.session_state.get('threshold_optimal', 80)}%)"
            )

    with st.container(border=True):
        st.subheader("Task Timeline by Employee")

        missing_dates = sum(
            1 for p in st.session_state.projects for s in p["subtasks"]
            if not s.get("assigned_date") or not s.get("due_date")
        ) + sum(
            1 for t in st.session_state.operational_tasks
            if not t.get("assigned_date") or not t.get("due_date")
        )
        if missing_dates:
            st.warning(f"{missing_dates} task(s) don't have a Start/Due Date on file yet, so they're missing from the chart below.")
            if st.button(f"Fill in default dates (today \u2192 +7 days) for these {missing_dates} task(s)"):
                updated = backfill_missing_dates()
                try:
                    save_data_to_workbook(DATA_FILE_PATH)
                    st.session_state.data_file_mtime = DATA_FILE_PATH.stat().st_mtime
                except Exception as exc:
                    st.warning(f"Updated in-app, but couldn't sync to file: {exc}")
                st.success(f"Set default dates for {updated} task(s).")
                st.rerun()

        timeline_df = build_task_timeline_df(st.session_state.projects, st.session_state.operational_tasks)
        if timeline_df.empty:
            st.caption(
                "No tasks with both a start and due date yet. Set a Due Date when assigning a "
                "task in the Talent Matching Engine tab to see it plotted here."
            )
        else:
            gantt_fig = px.timeline(
                timeline_df, x_start="Start", x_end="Finish", y="Employee", color="Status",
                hover_name="Task", hover_data={"Project": True, "Status": False},
                color_discrete_map={"Completed": "#2A9D8F", "In Progress": ACCENT_GREEN},
            )
            gantt_fig.update_yaxes(autorange="reversed")
            gantt_fig.add_vline(x=datetime.now(), line_dash="dash", line_color="#E63946")
            gantt_fig.update_layout(
                height=max(300, 45 * timeline_df["Employee"].nunique()),
                margin=dict(t=20, b=20), xaxis_title="", legend_title="",
            )
            st.plotly_chart(gantt_fig, use_container_width=True)
            st.caption("Red dashed line marks today. Each bar runs from when the task was assigned to its due date.")

with tab_matching:
    st.subheader("Talent & Capacity Matching Engine")
    st.caption(
        "Define the work, choose the domain and skills it needs, and the "
        "engine ranks every engineer in that domain by skill match (then by "
        "lowest utilization to prevent burnout), then assign in one step."
    )

    entry_type = st.radio(
        "What are you assigning?",
        ["Core Project Subtask", "Operational Task (SLA / Ad-hoc)"],
        horizontal=True,
    )

    input_col, result_col = st.columns([1, 1.4])

    with input_col:
        if entry_type == "Core Project Subtask":
            project_name = st.text_input(
                "Project Name",
                placeholder="e.g. Q4 Network Resilience Upgrade",
                help="Matches an existing project (case-insensitive) to add this "
                     "subtask to it; otherwise a new project is created.",
            )
            task_name = st.text_input("Subtask Name", placeholder="e.g. Site Survey")
        else:
            project_name = None
            task_name = st.text_input(
                "Task Name", placeholder="e.g. SLA Ticket: VPN Access Issue"
            )

        selected_domain = st.selectbox(
            "Domain", DOMAINS,
            help="Scopes both the skill list and the candidate pool to this domain."
        )
        task_effort = st.slider("Estimated Effort (Hours/Week)", 5, 25, 10)
        task_criticality = st.select_slider(
            "Task Criticality", options=[1, 2, 3, 4, 5], value=3,
            help="1 = low impact if delayed, 5 = mission-critical.",
        )
        due_date = st.date_input(
            "Due Date", value=date.today() + timedelta(days=7), min_value=date.today(),
            help="Used to build the Task Timeline chart on the Capacity Dashboard.",
        )
        required_skills = st.multiselect(
            "Skills Needed",
            options=SKILLS_BY_DOMAIN[selected_domain],
            help="Candidates are ranked by how many of these they have.",
        )
        enable_pairing = st.checkbox(
            "Enable Knowledge Sharing (Senior + PDP Pairing)",
            help="Pairs the best-ranked available Senior with the best-ranked "
                 "available PDP to distribute load and facilitate peer shadowing."
        )
        allow_overload = st.checkbox(
            "Override 100% capacity restriction",
            help="By default, candidates who would exceed 100% utilization are "
                 "excluded. Check this only if assigning the task anyway is necessary."
        )
        display_task_name = task_name.strip() if task_name.strip() else "Untitled Task"

    domain_pool = df[df["Primary Domain"] == selected_domain]
    candidates = rank_candidates_by_skills(domain_pool, required_skills)
    recommendation_ids = []
    senior_share = junior_share = None

    with result_col:
        st.markdown(
            f"**Task:** {display_task_name}  |  **Domain:** `{selected_domain}`  |  "
            f"**Candidates found:** {len(candidates)}"
        )

        if not candidates.empty:
            st.caption("Ranked candidates (best skill match, then lowest utilization):")
            st.dataframe(
                candidates[["Name", "Experience Level", "Skill Match", "Utilization %", "Status"]]
                .reset_index(drop=True),
                use_container_width=True, hide_index=True, height=200,
            )

        if not enable_pairing:
            required_pct = task_effort / STANDARD_WEEKLY_CAPACITY * 100
            safe_candidates = candidates[candidates["Utilization %"] + required_pct <= 100]
            default_pool = safe_candidates if not safe_candidates.empty else candidates

            if candidates.empty:
                st.error("No engineers found in this domain.")
            else:
                all_names = candidates["Name"].tolist()
                default_name = default_pool.iloc[0]["Name"] if not default_pool.empty else all_names[0]
                chosen_name = st.selectbox(
                    "Assign To", options=all_names,
                    index=all_names.index(default_name),
                    help="Defaults to the top-ranked match. Pick anyone else in the domain instead.",
                )
                chosen_row = candidates[candidates["Name"] == chosen_name].iloc[0]
                projected_util = round(chosen_row["Utilization %"] + required_pct, 1)

                if projected_util > 100 and not allow_overload:
                    st.error(
                        f"**{chosen_name}** would exceed 100% utilization ({projected_util}%). "
                        "Enable the override checkbox to assign anyway, or pick someone else."
                    )
                else:
                    recommendation_ids = [chosen_row["Employee ID"]]
                    skill_note = f", Skill Match {chosen_row['Skill Match']}/{len(required_skills)}" if required_skills else ""
                    st.success(f"**Assigning '{display_task_name}' to:** {chosen_name} ({chosen_row['Experience Level']}){skill_note}")
                    st.write(f"Current Utilization: **{chosen_row['Utilization %']}%**")
                    st.write(f"Projected Utilization if Assigned: **{projected_util}%**")
                    if projected_util > 100:
                        crit_note = " This is a high-criticality task, so getting it right matters more than usual." if task_criticality >= 4 else ""
                        st.warning(f"This assignment will push {chosen_name} into Burnout Risk "
                                   f"(allowed because the override is enabled).{crit_note}")

        else:
            senior_share = round(task_effort * 0.6, 1)
            junior_share = round(task_effort - senior_share, 1)
            senior_pct = senior_share / STANDARD_WEEKLY_CAPACITY * 100
            junior_pct = junior_share / STANDARD_WEEKLY_CAPACITY * 100

            seniors_all = candidates[candidates["Experience Level"] == "Senior"]
            juniors_all = candidates[candidates["Experience Level"] == "PDP"]
            seniors_safe = seniors_all[seniors_all["Utilization %"] + senior_pct <= 100]
            juniors_safe = juniors_all[juniors_all["Utilization %"] + junior_pct <= 100]
            seniors_default_pool = seniors_safe if not seniors_safe.empty else seniors_all
            juniors_default_pool = juniors_safe if not juniors_safe.empty else juniors_all

            if seniors_all.empty or juniors_all.empty:
                st.error(
                    "Pairing requires at least one available Senior AND one PDP "
                    "in this domain. Try disabling pairing or choosing another domain."
                )
            else:
                senior_names = seniors_all["Name"].tolist()
                junior_names = juniors_all["Name"].tolist()
                default_senior = seniors_default_pool.iloc[0]["Name"] if not seniors_default_pool.empty else senior_names[0]
                default_junior = juniors_default_pool.iloc[0]["Name"] if not juniors_default_pool.empty else junior_names[0]

                pick_col1, pick_col2 = st.columns(2)
                with pick_col1:
                    chosen_senior_name = st.selectbox(
                        "Senior (Lead)", options=senior_names, index=senior_names.index(default_senior),
                    )
                with pick_col2:
                    chosen_junior_name = st.selectbox(
                        "PDP (Shadow)", options=junior_names, index=junior_names.index(default_junior),
                    )

                top_senior = seniors_all[seniors_all["Name"] == chosen_senior_name].iloc[0]
                top_junior = juniors_all[juniors_all["Name"] == chosen_junior_name].iloc[0]
                proj_senior = round(top_senior["Utilization %"] + senior_pct, 1)
                proj_junior = round(top_junior["Utilization %"] + junior_pct, 1)
                either_over = proj_senior > 100 or proj_junior > 100

                if either_over and not allow_overload:
                    st.error(
                        f"This pair would push someone over 100% utilization "
                        f"(Senior \u2192 {proj_senior}%, Junior \u2192 {proj_junior}%). "
                        "Enable the override checkbox to assign anyway, or pick different people."
                    )
                else:
                    recommendation_ids = [top_senior["Employee ID"], top_junior["Employee ID"]]
                    st.success(f"**Assigning '{display_task_name}' (Knowledge Sharing) to:**")
                    st.write(f"Lead: **{top_senior['Name']}** (Senior, Skill Match {top_senior['Skill Match']}), +{senior_share} hrs \u2192 {proj_senior}%")
                    st.write(f"Shadow: **{top_junior['Name']}** (PDP, Skill Match {top_junior['Skill Match']}), +{junior_share} hrs \u2192 {proj_junior}%")
                    if either_over:
                        crit_note = " This is a high-criticality task, so getting it right matters more than usual." if task_criticality >= 4 else ""
                        st.warning(f"At least one person in this pair will exceed 100% utilization "
                                   f"(allowed because the override is enabled).{crit_note}")

        assign_clicked = st.button("Assign & Save", use_container_width=True)

        if assign_clicked:
            if not task_name.strip():
                st.warning("Enter a task/subtask name before assigning.")
            elif entry_type == "Core Project Subtask" and not project_name.strip():
                st.warning("Enter a project name before assigning.")
            elif not recommendation_ids:
                st.warning("No valid candidate to assign. Adjust the domain or pairing settings.")
            else:
                hours_per_person = []
                reassignment_notes = []
                for emp_id in recommendation_ids:
                    hrs = senior_share if (enable_pairing and emp_id == recommendation_ids[0]) else \
                        (junior_share if enable_pairing else task_effort)
                    person_row = st.session_state.roster.loc[st.session_state.roster["Employee ID"] == emp_id].iloc[0]
                    projected_for_person = person_row["Utilization %"] + (hrs / STANDARD_WEEKLY_CAPACITY * 100)
                    if task_criticality >= 4 and projected_for_person >= NEAR_FULL_THRESHOLD:
                        note = attempt_task_reassignment(person_row["Name"])
                        if note:
                            reassignment_notes.append(note)
                    add_hours(emp_id, hrs)
                    hours_per_person.append(hrs)

                names = ", ".join(
                    st.session_state.roster.loc[st.session_state.roster["Employee ID"] == eid, "Name"].values[0]
                    for eid in recommendation_ids
                )
                hours_str = ", ".join(f"{h:g}" for h in hours_per_person)
                skills_str = ", ".join(required_skills)

                if entry_type == "Core Project Subtask":
                    existing_project = next(
                        (p for p in st.session_state.projects
                         if p["project"].strip().lower() == project_name.strip().lower()),
                        None,
                    )
                    subtask_record = {
                        "subtask": task_name.strip(), "assigned_to": names,
                        "skills_needed": skills_str, "hours": hours_str, "completed": False,
                        "criticality": task_criticality,
                        "assigned_date": str(date.today()), "due_date": str(due_date),
                    }
                    if existing_project:
                        existing_project["subtasks"].append(subtask_record)
                    else:
                        st.session_state.projects.append({
                            "domain": selected_domain, "project": project_name.strip(), "subtasks": [subtask_record],
                        })
                else:
                    st.session_state.operational_tasks.append({
                        "task": task_name.strip(), "assigned_to": names,
                        "skills_needed": skills_str, "hours": hours_str, "completed": False,
                        "criticality": task_criticality,
                        "assigned_date": str(date.today()), "due_date": str(due_date),
                    })

                st.session_state.assignment_log.append(
                    f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] "
                    f"'{display_task_name}': {task_effort} hrs/week ({selected_domain}), "
                    f"criticality {task_criticality}/5, due {due_date} → {names}"
                )
                st.session_state.setdefault("structured_log", []).append({
                    "timestamp": datetime.now(), "action": "Assigned", "task_name": display_task_name,
                    "domain": selected_domain, "hours": task_effort, "assignees": names,
                    "criticality": task_criticality,
                })
                for note in reassignment_notes:
                    st.session_state.assignment_log.append(f"[{datetime.now().strftime('%Y-%m-%d %H:%M')}] {note}")
                st.success(f"Assigned **'{display_task_name}'** to **{names}**. See it in the Projects & Operational Tasks tab.")
                for note in reassignment_notes:
                    st.info(note)
                try:
                    save_data_to_workbook(DATA_FILE_PATH)
                    st.session_state.data_file_mtime = DATA_FILE_PATH.stat().st_mtime
                except Exception as exc:
                    st.warning(f"Assigned in-app, but couldn't sync to file: {exc}")
                st.rerun()


with tab_projects:
    st.subheader("Core Projects: Subtasks & Assignees")
    st.caption(
        "Every Core Project broken down into its subtasks, with who is currently "
        "assigned. Add new projects and tasks from the Talent Matching Engine tab. "
        "Mark a subtask complete to free up its hours and credit a milestone."
    )
    for p_idx, project in enumerate(st.session_state.projects):
        with st.expander(f"{project['project']} ({project['domain']})"):
            for s_idx, subtask in enumerate(project["subtasks"]):
                render_task_row(subtask, "subtask", f"subtask_{p_idx}_{s_idx}", can_confirm=True)
                st.markdown("---")

    st.divider()

    st.subheader("Operational Tasks: Not Tied to Any Project")
    st.caption("SLA tickets and ad-hoc duties outside Core Project work, and who is handling them.")
    for t_idx, task in enumerate(st.session_state.operational_tasks):
        render_task_row(task, "task", f"optask_{t_idx}", can_confirm=True)
        st.markdown("---")

with tab_recognition:
    st.subheader("Recognition & Milestone Hub")
    st.caption(
        "Surfaces completed deliverables and current balance status so managers "
        "can recognize consistent, sustainable performers, not just the busiest ones."
    )

    hub_df = df[[
        "Name", "Role", "Primary Domain", "Experience Level", "Skills",
        "Completed Milestones Count", "Utilization %", "Status"
    ]].sort_values("Completed Milestones Count", ascending=False).reset_index(drop=True)

    def highlight_status(val):
        color = STATUS_COLOR.get(val, "#FFFFFF")
        return f"background-color: {color}22; color: {color}; font-weight: 600;"

    st.dataframe(hub_df.style.map(highlight_status, subset=["Status"]), use_container_width=True, height=420)

    star = compute_recognition_spotlight(df)
    if star is not None:
        st.success(
            f"**Spotlight:** {star['Name']}, {star['Completed Milestones Count']} milestones "
            f"completed while maintaining a healthy {star['Utilization %']}% utilization."
        )

with tab_raw:
    st.subheader("Full Roster Data (raw)")
    st.dataframe(df, use_container_width=True)

    st.divider()
    st.subheader("Export for Power BI")
    st.caption(
        "Downloads an analytics-ready workbook: employees with Utilization % and "
        "Status already computed, Core Projects and Operational Tasks combined into "
        "one flat table, and a structured activity log. Open it in Power BI Desktop "
        "(Get Data > Excel) and build visuals from there. This produces the data file "
        "only; the visuals themselves are built in Power BI."
    )
    export_buffer = build_powerbi_export(
        df, st.session_state.projects, st.session_state.operational_tasks,
        st.session_state.get("structured_log", []),
    )
    st.download_button(
        "Download Power BI Export (.xlsx)",
        data=export_buffer,
        file_name=f"align_powerbi_export_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        type="primary",
    )