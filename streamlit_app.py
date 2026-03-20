"""LLM-driven Student Performance Indicator (SPI) Streamlit app."""

from __future__ import annotations

import time

import streamlit as st

from llm import generate_spi_recommendation


QUIZ_QUESTIONS = [
    {
        "question": "If 3 notebooks cost 90 rupees, what is the cost of 7 notebooks?",
        "options": ["180", "190", "200", "210"],
        "answer": "210",
    },
    {
        "question": "Find the next number: 5, 11, 23, 47, ?",
        "options": ["86", "90", "95", "96"],
        "answer": "95",
    },
    {
        "question": "A task takes 4 hours for 2 students. How many hours for 4 students at same rate?",
        "options": ["1", "2", "3", "4"],
        "answer": "2",
    },
    {
        "question": "Which is the odd one out?",
        "options": ["Triangle", "Square", "Circle", "Cube"],
        "answer": "Cube",
    },
    {
        "question": "If today is Friday, what day will it be after 10 days?",
        "options": ["Sunday", "Monday", "Tuesday", "Wednesday"],
        "answer": "Monday",
    },
]


RATING_MAP = {
    "Very Low": 1,
    "Low": 2,
    "Medium": 3,
    "High": 4,
    "Very High": 5,
}


def rating(label: str) -> int:
    return RATING_MAP[label]


def quiz_summary() -> tuple[int, int, float]:
    correct = 0
    answered = 0
    for i, q in enumerate(QUIZ_QUESTIONS):
        selected = st.session_state.get(f"quiz_{i}")
        if selected:
            answered += 1
        if selected == q["answer"]:
            correct += 1
    score = (correct / len(QUIZ_QUESTIONS)) * 100
    return answered, correct, score


def main() -> None:
    st.set_page_config(page_title="SPI LLM Analyzer", page_icon="🎓", layout="wide")
    st.title("🎓 Student Performance Indicator (LLM Edition)")
    st.caption(
        "Answer the questionnaire and quiz. The app sends your profile to an LLM for personalized recommendations."
    )

    if "quiz_started_at" not in st.session_state:
        st.session_state.quiz_started_at = time.time()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Academic Inputs")
        cgpa = st.number_input("Current CGPA", min_value=0.0, max_value=10.0, value=7.0, step=0.1)
        attendance = st.number_input("Attendance Percentage", min_value=0, max_value=100, value=80)
        backlogs = st.number_input("Current Backlogs", min_value=0, max_value=20, value=0)
        assignment_score = st.number_input("Average Assignment Score (%)", min_value=0, max_value=100, value=75)
        project_score = st.number_input("Project Score (%)", min_value=0, max_value=100, value=75)

    with col2:
        st.subheader("Behavioural Inputs")
        study_consistency = st.selectbox("Study Routine Consistency", options=list(RATING_MAP.keys()), index=2)
        class_engagement = st.selectbox("Class Engagement", options=list(RATING_MAP.keys()), index=2)
        stress = st.selectbox("Academic Stress Level", options=list(RATING_MAP.keys()), index=2)
        motivation = st.selectbox("Motivation to Improve", options=list(RATING_MAP.keys()), index=2)
        confidence = st.selectbox("Confidence in Solving Problems", options=list(RATING_MAP.keys()), index=2)
        sleep_hours = st.number_input("Sleep Hours per Day", min_value=3, max_value=12, value=7)

    st.subheader("Quick Logic Quiz")
    st.caption("Answer all 5 questions for better recommendation quality.")

    for i, q in enumerate(QUIZ_QUESTIONS):
        st.radio(
            f"Q{i + 1}. {q['question']}",
            options=q["options"],
            index=None,
            key=f"quiz_{i}",
        )

    if st.button("Generate LLM Recommendation", use_container_width=True):
        answered, correct, quiz_score = quiz_summary()
        if answered < len(QUIZ_QUESTIONS):
            st.warning("Please answer all quiz questions before generating recommendation.")
            st.stop()

        time_taken = round(time.time() - st.session_state.quiz_started_at, 2)

        profile = {
            "academic": {
                "cgpa": cgpa,
                "attendance_percent": attendance,
                "backlogs": backlogs,
                "assignment_score_percent": assignment_score,
                "project_score_percent": project_score,
            },
            "behavioural": {
                "study_consistency": rating(study_consistency),
                "class_engagement": rating(class_engagement),
                "academic_stress": rating(stress),
                "motivation": rating(motivation),
                "confidence": rating(confidence),
                "sleep_hours": sleep_hours,
            },
            "quiz": {
                "total_questions": len(QUIZ_QUESTIONS),
                "correct_answers": correct,
                "score_percent": round(quiz_score, 2),
                "time_taken_seconds": time_taken,
            },
        }

        st.subheader("Computed Snapshot")
        st.json(profile)

        with st.spinner("Calling LLM for SPI analysis..."):
            recommendation = generate_spi_recommendation(profile)

        st.subheader("LLM Recommendation")
        st.markdown(recommendation)


if __name__ == "__main__":
    main()