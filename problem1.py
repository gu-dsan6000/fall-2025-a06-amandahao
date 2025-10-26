#!/usr/bin/env python3
import os
import sys
import time
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_extract, count


def create_spark_session(master_url: str) -> SparkSession:
    """Create a Spark session optimized for cluster execution with S3 access."""
    spark = (
        SparkSession.builder
        .appName("SparkLogLevelDistribution")
        .master(master_url)
        .config("spark.executor.memory", "2g")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.cores", "1")
        .config("spark.cores.max", "3")
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")
        .config("spark.hadoop.fs.s3a.aws.credentials.provider", "com.amazonaws.auth.InstanceProfileCredentialsProvider")
        .config("spark.hadoop.fs.s3a.path.style.access", "true")
        .config("spark.hadoop.fs.s3a.fast.upload", "true")
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .config("spark.sql.adaptive.enabled", "true")
        .getOrCreate()
    )
    return spark

# function to complete problem 1
def run_log_analysis(spark: SparkSession, bucket: str):
    """Analyze log level distribution across all YARN container logs."""
    start_time = time.time()

    input_path = f"{bucket}/data/*/*" # asterisks read all subdirectories
    output_dir = "data/output"

    # read all text files from s3
    logs_df = spark.read.text(input_path)
    total_lines = logs_df.count()
    print(f"✅ Loaded {total_lines:,} total log lines")

    # extract log level data
    pattern = r"\b(INFO|WARN|ERROR|DEBUG)\b"
    logs_with_levels = logs_df.withColumn("log_level", regexp_extract(col("value"), pattern, 1))
    filtered_logs = logs_with_levels.filter(col("log_level") != "")
    matched_lines = filtered_logs.count()

    # count per log level
    counts_df = (
        filtered_logs.groupBy("log_level")
        .agg(count("*").alias("count"))
        .orderBy(col("count").desc())
    )
    # save csv
    counts_list = counts_df.collect()
    # counts_df.show(10)

    
    # sample logs
    sample_df = filtered_logs.sample(fraction=0.001).limit(10)
    sample_df.show(10, truncate=False)
    sample_list = sample_df.select(col("value").alias("log_entry"), col("log_level")).collect()

    # save count csv manually
    os.makedirs(output_dir, exist_ok=True)
    counts_file = os.path.join(output_dir, "problem1_counts.csv")
    with open(counts_file, "w") as f:
        f.write("log_level,count\n")
        for row in counts_list:
            f.write(f"{row['log_level']},{row['count']}\n")

    # save sample csv manually
    sample_file = os.path.join(output_dir, "problem1_sample.csv")
    with open(sample_file, "w") as f:
        f.write("log_entry,log_level\n")
        for row in sample_list:
            log_entry = str(row['log_entry']).replace('"', '""')
            f.write(f'"{log_entry}",{row["log_level"]}\n')

    # summary stats
    unique_levels = [row["log_level"] for row in counts_list]
    elapsed = time.time() - start_time

    summary_file = os.path.join(output_dir, "problem1_summary.txt")
    with open(summary_file, "w") as f:
        f.write(f"Total log lines processed: {total_lines:,}\n")
        f.write(f"Total lines with log levels: {matched_lines:,}\n")
        f.write(f"Unique log levels found: {len(unique_levels)}\n\n")
        f.write("Log level distribution:\n")
        for row in counts_list:
            pct = (row['count'] / matched_lines) * 100
            f.write(f"  {row['log_level']:<6}: {row['count']:>10,} ({pct:5.2f}%)\n")
        f.write(f"\nProcessing time: {elapsed:.2f} seconds\n")

    print("\n" + "=" * 70)
    print("✅ Log Level Distribution Analysis Complete!")
    print(f"Output directory: {output_dir}")
    print(f"  - {counts_file}")
    print(f"  - {sample_file}")
    print(f"  - {summary_file}")
    print("=" * 70)


# ------------------------------------------------------------
# Main Entry Point
# ------------------------------------------------------------
def main():
    print("=" * 70)
    print("SPARK LOG LEVEL DISTRIBUTION (CLUSTER MODE)")
    print("=" * 70)

    if len(sys.argv) > 1:
        master_url = sys.argv[1]
    else:
        master_private_ip = os.getenv("MASTER_PRIVATE_IP")
        if master_private_ip:
            master_url = f"spark://{master_private_ip}:7077"
        else:
            print("❌ Error: Master URL not provided")
            return 1

    # get bucket
    bucket = os.getenv("SPARK_LOGS_BUCKET", "s3a://alh326-assignment-spark-cluster-logs")
    spark = create_spark_session(master_url)

    try:
        run_log_analysis(spark, bucket)
        success = True
    except Exception as e:
        print(f"❌ Error during log analysis: {e}")
        success = False
    finally:
        spark.stop()

    print("\n" + "=" * 70)
    if success:
        print("✅ Problem 1 Completed Successfully!")
        print("Next steps:")
        print("  1. Download output CSVs from master node")
        print("  2. Verify counts and summary files")
    else:
        print("❌ Problem 1 Failed - check logs for details.")
    print("=" * 70)

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())