import sqlite3
import uuid
from datetime import datetime

import pandas as pd
import streamlit as st


DB_PATH = "doodle_clone.db"


def get_conn():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def init_db():
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS polls (
            poll_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            description TEXT,
            creator_name TEXT,
            created_at TEXT NOT NULL
        )
        """
    )

    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS options (
            option_id TEXT PRIMARY KEY,
            poll_id TEXT NOT NULL,
            option_label TEXT NOT NULL,
            option_order INTEGER NOT NULL,
            FOREIGN KEY (poll_id) REFERENCES polls(poll_id) ON DELETE CASCADE
        )
        """
    )

    cur.execute(
        """
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
        """
    )

    conn.commit()
    conn.close()


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
    conn.close()
    return poll_id


def get_all_polls():
    conn = get_conn()
    query = """
        SELECT poll_id, title, description, creator_name, created_at
        FROM polls
        ORDER BY created_at DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def get_poll(poll_id):
    conn = get_conn()
    cur = conn.cursor()

    cur.execute(
        """
        SELECT poll_id, title, description, creator_name, created_at
        FROM polls
        WHERE poll_id = ?
        """,
        (poll_id,),
    )
    row = cur.fetchone()
    conn.close()

    if not row:
        return None

    return {
        "poll_id": row[0],
        "title": row[1],
        "description": row[2],
        "creator_name": row[3],
        "created_at": row[4],
    }


def get_poll_options(poll_id):
    conn = get_conn()
    query = """
        SELECT option_id, poll_id, option_label, option_order
        FROM options
        WHERE poll_id = ?
        ORDER BY option_order ASC
    """
    df = pd.read_sql_query(query, conn, params=(poll_id,))
    conn.close()
    return df


def get_poll_votes(poll_id):
    conn = get_conn()
    query = """
        SELECT vote_id, poll_id, voter_name, option_id, availability, updated_at
        FROM votes
        WHERE poll_id = ?
    """
    df = pd.read_sql_query(query, conn, params=(poll_id,))
    conn.close()
    return df


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
    conn.close()


def build_results_matrix(poll_id):
    options_df = get_poll_options(poll_id)
    votes_df = get_poll_votes(poll_id)

    if options_df.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    if votes_df.empty:
        summary_df = options_df[["option_label"]].copy()
        summary_df["yes"] = 0
        summary_df["maybe"] = 0
        summary_df["no"] = 0
        summary_df["score"] = 0
        return summary_df, pd.DataFrame(), options_df

    merged = votes_df.merge(options_df, on="option_id", how="left")

    summary = (
        merged.groupby("option_label")["availability"]
        .value_counts()
        .unstack(fill_value=0)
        .reset_index()
    )

    for col in ["yes", "maybe", "no"]:
        if col not in summary.columns:
            summary[col] = 0

    summary["score"] = summary["yes"] * 2 + summary["maybe"] * 1
    summary = summary[["option_label", "yes", "maybe", "no", "score"]].sort_values(
        by=["score", "yes", "maybe"], ascending=False
    )

    voter_matrix = merged.pivot_table(
        index="voter_name",
        columns="option_label",
        values="availability",
        aggfunc="first",
    )

    return summary, voter_matrix, options_df


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
        page = st.radio("Go to", ["Create Poll", "Open Poll", "Browse Polls"], index=0)

        if poll_id_from_url:
            st.info(f"Poll ID from URL: {poll_id_from_url}")
            if st.button("Open URL Poll"):
                page = "Open Poll"
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
                st.write("Or open it directly from the Open Poll page.")

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

            options_df = get_poll_options(poll_id)
            votes_df = get_poll_votes(poll_id)

            st.divider()
            st.markdown("### Submit or update your availability")

            existing_names = []
            if not votes_df.empty:
                existing_names = sorted(votes_df["voter_name"].dropna().unique().tolist())

            with st.form("vote_form"):
                voter_name = st.text_input("Your name", placeholder="Your name")
                st.caption("Choose Yes, Maybe, or No for each option.")

                selections = {}
                for _, row in options_df.iterrows():
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

            summary_df, voter_matrix, _ = build_results_matrix(poll_id)

            if not summary_df.empty:
                st.markdown("#### Option summary")
                st.dataframe(summary_df, use_container_width=True)

                best_score = summary_df["score"].max()
                best_options = summary_df.loc[summary_df["score"] == best_score, "option_label"].tolist()
                if best_options:
                    st.success("Best option(s): " + " | ".join(best_options))

            if not voter_matrix.empty:
                st.markdown("#### Voter matrix")
                display_matrix = voter_matrix.fillna("")
                st.dataframe(display_matrix, use_container_width=True)

            if existing_names:
                st.markdown("#### Current voters")
                st.write(", ".join(existing_names))

    elif page == "Browse Polls":
        st.subheader("All polls")
        polls_df = get_all_polls()

        if polls_df.empty:
            st.info("No polls found yet.")
        else:
            st.dataframe(polls_df, use_container_width=True)

            st.markdown("### Open a poll")
            selected_poll_id = st.selectbox(
                "Choose poll ID",
                options=polls_df["poll_id"].tolist(),
            )

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
