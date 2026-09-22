from dotenv import load_dotenv

from vector_store import load_vector_store

from agents import (
    diagnosis_agent,
    parts_agent,
    action_agent
)


load_dotenv()


# -----------------------------
# CONFIGURATION
# -----------------------------

CONFIDENCE_THRESHOLD = 0.25


# -----------------------------
# LOAD KNOWLEDGE BASE
# -----------------------------

print("Loading knowledge base...")

vector_store = load_vector_store()


# -----------------------------
# MAIN APPLICATION
# -----------------------------

def main():

    print("\n================================")
    print("FIELD SERVICE TROUBLESHOOTING")
    print("AI ASSISTANT")
    print("================================\n")

    question = input(
        "Describe the equipment problem:\n> "
    )

    # -----------------------------
    # RETRIEVAL
    # -----------------------------

    print("\nSearching equipment manual...")

    results = vector_store.similarity_search_with_relevance_scores(
        question,
        k=3
    )

    print(f"Retrieved {len(results)} relevant chunks")

    # Get strongest retrieval score
    best_score = max(
        score for document, score in results
    )

    print(f"Best relevance score: {best_score:.3f}")

    # -----------------------------
    # HUMAN ESCALATION
    # -----------------------------

    if best_score < CONFIDENCE_THRESHOLD:

        print("\n================================")
        print("HUMAN ESCALATION REQUIRED")
        print("================================\n")

        print(
            "The equipment manual does not contain "
            "sufficiently relevant information."
        )

        print(
            "\nRecommended Action:"
            "\nEscalate this issue to a human service engineer."
        )

        return

    # -----------------------------
    # BUILD CONTEXT
    # -----------------------------

    documents = [
        document
        for document, score in results
    ]

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    # -----------------------------
    # AGENT 1: DIAGNOSIS
    # -----------------------------

    print("\nRunning Diagnosis Agent...")

    diagnosis = diagnosis_agent(
        context,
        question
    )

    print("\nDiagnosis completed.")

    # -----------------------------
    # AGENT 2: PARTS
    # -----------------------------

    print("\nRunning Parts Recommendation Agent...")

    parts = parts_agent(
        context,
        diagnosis
    )

    print("Parts recommendation completed.")

    # -----------------------------
    # AGENT 3: NEXT ACTION
    # -----------------------------

    print("\nRunning Next-Best-Action Agent...")

    actions = action_agent(
        context,
        diagnosis,
        parts
    )

    print("Action planning completed.")

    # -----------------------------
    # FINAL OUTPUT
    # -----------------------------

    print("\n================================")
    print("AGENTIC TROUBLESHOOTING RESULT")
    print("================================")

    print("\n--- DIAGNOSIS ---")
    print(diagnosis)

    print("\n--- RELEVANT PARTS ---")
    print(parts)

    print("\n--- NEXT-BEST ACTION ---")
    print(actions)

    print("\n================================")
    print("Human Escalation: NO")
    print("================================")


if __name__ == "__main__":
    main()