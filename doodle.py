import sqlite3
import uuid
from datetime import datetime

import streamlit as st

DB_PATH = "doodle_clone.db"


@st.cache_resource
def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


@st.cache_resource
def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS polls (
            poll_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            creator_name TEXT,
            created_at TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS options (
            option_id TEXT PRIMARY KEY,
            poll_id TEXT NOT NULL,
            option_label TEXT NOT NULL,
            option_order INTEGER NOT NULL,
            FOREIGN KEY (poll_id) REFERENCES polls(poll_id) ON DELETE CASCADE
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS votes (
            vote_id TEXT PRIMARY KEY,
            poll_id TEXT NOT NULL,
            voter_name TEXT NOT NULL,
            option_id TEXT NOT NULL,
            availability TEXT NOT NULL CHECK (availability IN ('yes', 'maybe', 'no')),
            updated_at TEXT NOT NULL,
            UNIQUE(poll_id, voter_name, option_id),
            FOREIGN KEY (poll_id) REFERENCES polls(poll_id) ON DELETE CASCADE,
            FOREIGN KEY (option_id) REFERENCES options(option_id) ON DELETE CASCADE
        )
    """)

    conn.commit()


def create_poll(title, description, creator_name, options):
    poll_id = uuid.uuid4().hex[:10]
    created_at = datetime.utcnow().isoformat()
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        INSERT INTO polls (poll_id, title, description, creator_name, created_at)
        VALUES (?, ?, ?, ?, ?)
        """,
        (poll_id, title.strip(), description.strip(), creator_name.strip(), created_at),
    )

    for idx, option in enumerate(options):
        cur.execute(
            """
            INSERT INTO options (option_id, poll_id, option_label, option_order)
            VALUES (?, ?, ?, ?)
            """,
            (uuid.uuid4().hex, poll_id, option.strip(), idx),
        )

    conn.commit()
    return poll_id


def get_all_polls():
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT poll_id, title, description, creator_name, created_at
        FROM polls
        ORDER BY created_at DESC
    """)
    return [dict(row) for row in cur.fetchall()]


def get_poll(poll_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT poll_id, title, description, creator_name, created_at
        FROM polls
        WHERE poll_id = ?
    """, (poll_id,))
    row = cur.fetchone()
    return dict(row) if row else None


def get_poll_options(poll_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT option_id, poll_id, option_label, option_order
        FROM options
        WHERE poll_id = ?
        ORDER BY option_order ASC
    """, (poll_id,))
    return [dict(row) for row in cur.fetchall()]


def get_poll_votes(poll_id):
    conn = get_conn()
    cur = conn.cursor()
    cur.execute("""
        SELECT vote_id, poll_id, voter_name, option_id, availability, updated_at
        FROM votes
        WHERE poll_id = ?
    """, (poll_id,))
    return [dict(row) for row in cur.fetchall()]


def save_votes(poll_id, voter_name, selections):
    voter_name = voter_name.strip()
    if not voter_name:
        raise ValueError("Voter name is required.")

    conn = get_conn()
    cur = conn.cursor()
    now = datetime.utcnow().isoformat()

    for option_id, availability in selections.items():
        cur.execute(
            """
            INSERT INTO votes (vote_id, poll_id, voter_name, option_id, availability, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(poll_id, voter_name, option_id)
            DO UPDATE SET
                availability = excluded.availability,
                updated_at = excluded.updated_at
            """,
            (uuid.uuid4().hex, poll_id, voter_name, option_id, availability, now),
        )

    conn.commit()


def build_results_matrix(poll_id):
    options = get_poll_options(poll_id)
    votes = get_poll_votes(poll_id)

    if not options:
        return [], [], []

    option_lookup = {o["option_id"]: o["option_label"] for o in options}
    summary = {
        o["option_label"]: {"option_label": o["option_label"], "yes": 0, "maybe": 0, "no": 0, "score": 0}
        for o in options
    }

    voter_matrix = {}

    for vote in votes:
        option_label = option_lookup.get(vote["option_id"])
        if not option_label:
            continue

        avail = vote["availability"]
        voter = vote["voter_name"]

        if avail in ("yes", "maybe", "no"):
            summary[option_label][avail] += 1

        if voter not in voter_matrix:
            voter_matrix[voter] = {}
        voter_matrix[voter][option_label] = avail

    for option_label in summary:
        summary[option_label]["score"] = summary[option_label]["yes"] * 2 + summary[option_label]["maybe"]

    summary_rows = sorted(
        summary.values(),
        key=lambda x: (x["score"], x["yes"], x["maybe"]),
        reverse=True,
    )

    ordered_option_labels = [o["option_label"] for o in options]
    voter_rows = []
    for voter_name in sorted(voter_matrix.keys()):
        row = {"voter_name": voter_name}
        for label in ordered_option_labels:
            row[label] = voter_matrix[voter_name].get(label, "")
        voter_rows.append(row)

    return summary_rows, voter_rows, options


def get_poll_url_hint(poll_id):
    return f"?poll_id={poll_id}"


def main():
    st.set_page_config(page_title="Simple Doodle Clone", layout="wide")
    init_db()

    st.title("Simple Doodle-Style Poll App")
    st.caption("Create a poll, share the poll ID, and collect availability votes.")

    query_params = st.query_params
    poll_id_from_url = query_params.get("poll_id", "")

    with st.sidebar:
        st.header("Navigation")
        pages = ["Create Poll", "Open Poll", "Browse Polls"]
        default_index = 1 if poll_id_from_url else 0
        page = st.radio("Go to", pages, index=default_index)

        if poll_id_from_url:
            st.info(f"Poll ID from URL: {poll_id_from_url}")
            st.session_state["poll_id_input"] = poll_id_from_url

    if page == "Create Poll":
        st.subheader("Create a new poll")

        with st.form("create_poll_form"):
            title = st.text_input("Poll title", placeholder="Team meeting availability")
            description = st.text_area(
                "Description",
                placeholder="Pick all times that work for you.",
                height=120,
            )
            creator_name = st.text_input("Your name", placeholder="Shayne")

            st.markdown("**Enter date/time options (one per line)**")
            raw_options = st.text_area(
                "Options",
                placeholder="2026-03-25 10:00 AM\n2026-03-25 2:00 PM\n2026-03-26 9:00 AM",
                height=180,
            )

            submitted = st.form_submit_button("Create Poll")

        if submitted:
            options = [x.strip() for x in raw_options.splitlines() if x.strip()]
            if not title.strip():
                st.error("Poll title is required.")
            elif len(options) < 2:
                st.error("Please provide at least two options.")
            else:
                new_poll_id = create_poll(title, description, creator_name, options)
                st.success(f"Poll created: {new_poll_id}")
                st.code(new_poll_id)
                st.write("Share this URL pattern with the poll ID appended:")
                st.code(get_poll_url_hint(new_poll_id))

    elif page == "Open Poll":
        st.subheader("Open and vote on a poll")

        default_poll_id = st.session_state.get("poll_id_input", poll_id_from_url)
        poll_id = st.text_input("Enter poll ID", value=default_poll_id)

        if poll_id:
            poll = get_poll(poll_id)
            if not poll:
                st.error("Poll not found.")
                return

            st.markdown(f"## {poll['title']}")
            if poll["description"]:
                st.write(poll["description"])

            meta_cols = st.columns(3)
            meta_cols[0].write(f"**Poll ID:** {poll['poll_id']}")
            meta_cols[1].write(f"**Creator:** {poll['creator_name'] or 'Unknown'}")
            meta_cols[2].write(f"**Created:** {poll['created_at'][:19].replace('T', ' ')}")

            options = get_poll_options(poll_id)
            votes = get_poll_votes(poll_id)

            st.divider()
            st.markdown("### Submit or update your availability")

            existing_names = sorted({v["voter_name"] for v in votes if v["voter_name"]})

            with st.form("vote_form"):
                voter_name = st.text_input("Your name", placeholder="Your name")
                st.caption("Choose Yes, Maybe, or No for each option.")

                selections = {}
                for row in options:
                    option_id = row["option_id"]
                    option_label = row["option_label"]
                    selections[option_id] = st.radio(
                        option_label,
                        options=["yes", "maybe", "no"],
                        index=0,
                        horizontal=True,
                        key=f"vote_{option_id}",
                    )

                vote_submit = st.form_submit_button("Save my votes")

            if vote_submit:
                try:
                    save_votes(poll_id, voter_name, selections)
                    st.success("Votes saved.")
                    st.rerun()
                except ValueError as e:
                    st.error(str(e))

            st.divider()
            st.markdown("### Results")

            summary_rows, voter_rows, _ = build_results_matrix(poll_id)

            if summary_rows:
                st.markdown("#### Option summary")
                st.dataframe(summary_rows, use_container_width=True)

                best_score = max(row["score"] for row in summary_rows)
                best_options = [row["option_label"] for row in summary_rows if row["score"] == best_score]
                if best_options:
                    st.success("Best option(s): " + " | ".join(best_options))

            if voter_rows:
                st.markdown("#### Voter matrix")
                st.dataframe(voter_rows, use_container_width=True)

            if existing_names:
                st.markdown("#### Current voters")
                st.write(", ".join(existing_names))

    elif page == "Browse Polls":
        st.subheader("All polls")
        polls = get_all_polls()

        if not polls:
            st.info("No polls found yet.")
        else:
            st.dataframe(polls, use_container_width=True)

            st.markdown("### Open a poll")
            poll_ids = [p["poll_id"] for p in polls]
            selected_poll_id = st.selectbox("Choose poll ID", options=poll_ids)

            if selected_poll_id:
                poll = get_poll(selected_poll_id)
                if poll:
                    st.markdown(f"**Title:** {poll['title']}")
                    st.markdown(f"**Description:** {poll['description'] or ''}")
                    st.markdown(f"**Creator:** {poll['creator_name'] or 'Unknown'}")
                    st.markdown(f"**Created:** {poll['created_at'][:19].replace('T', ' ')}")
                    st.code(get_poll_url_hint(selected_poll_id))


if __name__ == "__main__":
    main()
