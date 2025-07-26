import json
import os
import statistics
from pathlib import Path
from collections import defaultdict

output_folder = "enem"
def process_eval_scores(json_file="eval_enem_questions.json"):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for item in data:
            scores = [
                metric["score"]
                for metric in item.get("metrics", [])
                if "score" in metric
            ]

            if scores:
                item["average_score"] = statistics.mean(scores)
            else:
                item["average_score"] = 0.0

        with open(json_file, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        return data

    except Exception as e:
        print(f"Error processing evaluation scores: {e}")
        return None


def group_by_score_range(data, step=0.1):
    groups = defaultdict(list)

    for item in data:
        score = item.get("average_score", 0.0)
        lower_bound = int(score / step) * step
        upper_bound = lower_bound + step

        range_key = f"{lower_bound:.1f} ~ {upper_bound:.1f}"

        groups[range_key].append(item["index"])

    return dict(sorted(groups.items()))


def calculate_metric_statistics(data):
    metric_scores = defaultdict(list)

    for item in data:
        for metric in item.get("metrics", []):
            if "metric_name" in metric and "score" in metric:
                metric_scores[metric["metric_name"]].append(metric["score"])

    metric_extremes = {}
    for metric_name in metric_scores:
        highest_score = -float("inf")
        lowest_score = float("inf")
        highest_item = None
        lowest_item = None

        for item in data:
            for metric in item.get("metrics", []):
                if metric.get("metric_name") == metric_name and "score" in metric:
                    score = metric["score"]
                    if score > highest_score:
                        highest_score = score
                        highest_item = item["index"]
                    if score < lowest_score:
                        lowest_score = score
                        lowest_item = item["index"]

        metric_extremes[metric_name] = {
            "highest": {"score": highest_score, "item": highest_item},
            "lowest": {"score": lowest_score, "item": lowest_item},
        }

        for item in data:
            item["metric_extremes"] = metric_extremes

    metric_stats = {}
    for metric_name, scores in metric_scores.items():
        average = statistics.mean(scores) if scores else 0.0
        metric_stats[metric_name] = {
            "average": average,
            "extremes": {
                "highest": {
                    "score": metric_extremes[metric_name]["highest"]["score"],
                    "item_index": metric_extremes[metric_name]["highest"]["item"],
                },
                "lowest": {
                    "score": metric_extremes[metric_name]["lowest"]["score"],
                    "item_index": metric_extremes[metric_name]["lowest"]["item"],
                },
            },
        }

    sorted_metrics = {
        k: v
        for k, v in sorted(
            metric_stats.items(), key=lambda item: item[1]["average"], reverse=True
        )
    }

    return sorted_metrics


if __name__ == "__main__":
    json_file = os.path.join(f"eval_result/{output_folder}", f"eval_{output_folder}_questions.json")
    updated_data = process_eval_scores(json_file=json_file)

    if updated_data:
        print(f"Processed {len(updated_data)} evaluation items")

        grouped_data = group_by_score_range(updated_data)

        print("Data grouped by score range:")
        for range_key, indices in grouped_data.items():
            print(f"{range_key}: {indices}")

        metric_averages = calculate_metric_statistics(updated_data)
        print("Average scores by metric:")
        if metric_averages:
            print("\nMetric Statistics Tree:")
            print("------------------------")
            for metric, stats in metric_averages.items():
                print(f"- {metric}")
                print(f"  - Average: {stats['average']:.2f}")
                print(f"  - Highest:")
                print(f"    - Score: {stats['extremes']['highest']['score']:.2f}")
                print(f"    - Index: {stats['extremes']['highest']['item_index']}")
                print(f"  - Lowest:")
                print(f"    - Score: {stats['extremes']['lowest']['score']:.2f}")
                print(f"    - Index: {stats['extremes']['lowest']['item_index']}")
                print()
        else:
            print("No metric data available")
        
        # Save grouped data and metric averages to JSON
        output_dir = Path(f"eval_result/{output_folder}/analysis")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save grouped data
        grouped_data_file = output_dir / "grouped_scores.json"
        with open(grouped_data_file, "w", encoding="utf-8") as f:
            json.dump(grouped_data, f, indent=2, ensure_ascii=False)
        print(f"Grouped data saved to {grouped_data_file}")

        # Save metric averages
        metric_stats_file = output_dir / "metric_statistics.json"
        with open(metric_stats_file, "w", encoding="utf-8") as f:
            json.dump(metric_averages, f, indent=2, ensure_ascii=False)
        print(f"Metric statistics saved to {metric_stats_file}")
