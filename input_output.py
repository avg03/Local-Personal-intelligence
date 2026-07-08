from agents.agents import graph


def ingest_pdf():
    pdf_path = input("\nEnter PDF path: ").strip()

    state = {
        "pdf_path": pdf_path,
        "query": "",

        "logs": [],
        "response": {},

        "similarity_threshold": 0.40,
    }

    result = graph.invoke(state)

    print("\n========== INGESTION COMPLETE ==========")

    for log in result.get("logs", []):
        print(f"• {log}")


def ask_query():
    query = input("\nAsk your question: ").strip()

    state = {
        "pdf_path": None,
        "query": query,

        "logs": [],
        "response": {},

        "similarity_threshold": 0.40,
    }

    result = graph.invoke(state)

    print("\n========== ANSWER ==========\n")
    print(result["response"]["answer"])

    print("\n========== SOURCES ==========\n")

    for source in result["response"].get("sources", []):
        print(f"Resource : {source['resource_name']}")
        print(f"Path     : {source['resource_path']}")
        print("-" * 50)


def main():

    while True:

        print("\n========== Local Personal Intelligence ==========")
        print("1. Ingest PDF")
        print("2. Ask Question")
        print("3. Exit")

        choice = input("\nChoice: ").strip()

        if choice == "1":
            ingest_pdf()

        elif choice == "2":
            ask_query()

        elif choice == "3":
            print("Goodbye!")
            break

        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()