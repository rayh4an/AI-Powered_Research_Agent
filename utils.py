from datetime import datetime
import os


def save_report(report: str, query: str):
    """Saves the final report to a markdown file."""

    # Create reports folder if it doesn't exist
    os.makedirs("reports", exist_ok=True)

    # Create a filename from the query and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_query = query[:30].replace(" ", "_").replace("?", "")
    filename = f"reports/{timestamp}_{safe_query}.md"

    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"# Research Report\n\n")
        f.write(f"**Query:** {query}\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        f.write(report)

    print(f"\n[SAVED] Report saved to: {filename}")
    return filename