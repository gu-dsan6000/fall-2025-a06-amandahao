#!/usr/bin/env python3
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, regexp_extract, try_to_timestamp, min as spark_min,
    max as spark_max, count as spark_count, when, lit
)

OUTPUT_DIR = "data/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def create_spark_session(master_url: str):
    spark = (
        SparkSession.builder
        .appName("ClusterUsageAnalysis")
        .master(master_url)
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.cores", "1")
        .config("spark.cores.max", "3")
        .getOrCreate()
    )
    return spark

def run_spark_analysis(spark, bucket):
    input_path = f"{bucket}/data/*/*"
    logs_df = spark.read.text(input_path)

    # extract timestamp
    ts_pattern = r"^(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2})"
    df = logs_df.withColumn("timestamp_raw", regexp_extract(col("value"), ts_pattern, 1))

    # extract task/stage/TID
    task_pattern = r"Running task ([\d\.]+) in stage ([\d\.]+) \(TID (\d+)\)"
    df = df.withColumn("app_number", regexp_extract(col("value"), task_pattern, 1))
    df = df.withColumn("cluster_id", regexp_extract(col("value"), task_pattern, 2))
    df = df.withColumn("application_id", regexp_extract(col("value"), task_pattern, 3))

    # parse timestamps
    df = df.withColumn(
        "timestamp",
        try_to_timestamp(col("timestamp_raw"), lit("yy/MM/dd HH:mm:ss"))
    )

    # skip any bad lines
    df_valid = df.filter(
        col("timestamp").isNotNull() &
        col("cluster_id").isNotNull() &
        col("application_id").isNotNull() &
        col("app_number").isNotNull()
    )

    # debugging: check if any valid rows
    if df_valid.count() == 0:
        print("⚠️ parsing error")
        return pd.DataFrame(), pd.DataFrame()

    # extract start/end time per application
    timeline_df = (
        df_valid.groupBy("cluster_id", "application_id", "app_number")
        .agg(
            spark_min("timestamp").alias("start_time"),
            spark_max("timestamp").alias("end_time")
        )
        .orderBy("cluster_id", "start_time")
    )

    # aggregated cluster statistics
    cluster_summary_df = (
        timeline_df.groupBy("cluster_id")
        .agg(
            spark_count("application_id").alias("num_applications"),
            spark_min("start_time").alias("cluster_first_app"),
            spark_max("end_time").alias("cluster_last_app")
        )
        .orderBy(col("num_applications").desc())
    )

    # save outputs
    timeline_pd = timeline_df.toPandas()
    cluster_summary_pd = cluster_summary_df.toPandas()

    timeline_pd.to_csv(os.path.join(OUTPUT_DIR, "problem2_timeline.csv"), index=False)
    cluster_summary_pd.to_csv(os.path.join(OUTPUT_DIR, "problem2_cluster_summary.csv"), index=False)

    # summary stats
    total_clusters = cluster_summary_pd.shape[0]
    total_apps = timeline_pd.shape[0]
    avg_apps = total_apps / total_clusters if total_clusters > 0 else 0
    most_used = cluster_summary_pd.sort_values("num_applications", ascending=False)

    stats_file = os.path.join(OUTPUT_DIR, "problem2_stats.txt")
    with open(stats_file, "w") as f:
        f.write(f"Total unique clusters: {total_clusters}\n")
        f.write(f"Total applications: {total_apps}\n")
        f.write(f"Average applications per cluster: {avg_apps:.2f}\n\n")
        f.write("Most heavily used clusters:\n")
        for _, row in most_used.iterrows():
            f.write(f"  Cluster {row['cluster_id']}: {row['num_applications']} applications\n")

    return timeline_pd, cluster_summary_pd

def generate_visualizations(timeline_pd, cluster_summary_pd):
    # debugging: check for empty dataframes
    if cluster_summary_pd.empty or timeline_pd.empty:
        print("⚠️ empty dataframes")
        return

    # BAR CHART
    plt.figure(figsize=(10,6))
    sns.barplot(data=cluster_summary_pd, x="cluster_id", y="num_applications", palette="viridis")
    plt.title("Applications per Cluster")
    plt.xlabel("Cluster ID")
    plt.ylabel("Number of Applications")
    for index, row in cluster_summary_pd.iterrows():
        plt.text(index, row.num_applications, str(row.num_applications), ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "problem2_bar_chart.png"))
    plt.close()

    # DENSITY PLOT
    largest_cluster = cluster_summary_pd.sort_values("num_applications", ascending=False).iloc[0]['cluster_id']
    timeline_pd['duration_sec'] = (
        pd.to_datetime(timeline_pd['end_time'], errors='coerce') -
        pd.to_datetime(timeline_pd['start_time'], errors='coerce')
    ).dt.total_seconds()

    largest_df = timeline_pd[
        (timeline_pd['cluster_id'] == str(largest_cluster)) &
        (timeline_pd['duration_sec'].notna()) &
        (timeline_pd['duration_sec'] > 0)
    ]

    # debugging: check for empty largest_df
    if largest_df.empty:
        print("⚠️ job duration empty")
        return

    plt.figure(figsize=(10,6))
    sns.histplot(largest_df['duration_sec'], kde=True, log_scale=True)
    plt.title(f"Job Duration Distribution for Cluster {largest_cluster} (n={largest_df.shape[0]})")
    plt.xlabel("Duration (seconds, log scale)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "problem2_density_plot.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("master_url", nargs="?", help="Spark master URL")
    parser.add_argument("--skip-spark", action="store_true", help="Skip Spark and read existing CSVs")
    parser.add_argument("--net-id", help="Network ID (for assignment runner)")
    args = parser.parse_args()

    if args.skip_spark:
        timeline_pd = pd.read_csv(os.path.join(OUTPUT_DIR, "problem2_timeline.csv"))
        cluster_summary_pd = pd.read_csv(os.path.join(OUTPUT_DIR, "problem2_cluster_summary.csv"))
    else:
        master_url = args.master_url or f"spark://{os.getenv('MASTER_PRIVATE_IP')}:7077"
        bucket = os.getenv("SPARK_LOGS_BUCKET", "s3a://alh326-assignment-spark-cluster-logs")
        spark = create_spark_session(master_url)
        try:
            timeline_pd, cluster_summary_pd = run_spark_analysis(spark, bucket)
        finally:
            spark.stop()

    generate_visualizations(timeline_pd, cluster_summary_pd)

if __name__ == "__main__":
    main()