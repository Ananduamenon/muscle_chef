from pyspark.sql import SparkSession, Window
import pandas as pd
from pyspark.sql.functions import sum as _sum, count, col, size, collect_set, rank, when, lit, min as _min

# create sparksession
spark = SparkSession.builder.appName("Muscle_chef_analysis").getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("ERROR")

# Read source
pandas_customer_df = pd.read_excel("Data/Customer.xls")
cust_df = spark.createDataFrame(pandas_customer_df)
shopping_df = spark.read.json("Data/Shipping.json", multiLine=True)
order_df = spark.read.option("InferSchema", "True").option("Header", "True").csv("Data/Order.csv")

country_order_join = cust_df.join(order_df, ["Customer_ID"], "inner").cache()

print("the total amount spent and the count for the Pending delivery status for each country.")

total_sale_by_country_df = country_order_join \
    .select("country", "Amount") \
    .groupBy("Country") \
    .agg(_sum("Amount").alias("Amount_spent_by_country"))

total_pending_status_by_country = cust_df.join(shopping_df.filter(col("Status") == "Pending"),
                                               cust_df.Customer_ID == shopping_df.Customer_ID, "inner") \
    .select("country", "Status") \
    .groupBy("Country") \
    .agg(count("Status").alias("Amount_spent_by_country"))

total_country_df = total_sale_by_country_df.join(total_pending_status_by_country, ["Country"], "outer").show()

print(
    "the total number of transactions, total quantity sold, and total amount spent for each customer, along with the product details.")

window = Window.partitionBy("Customer_ID").rowsBetween(Window.unboundedPreceding, Window.unboundedFollowing)

customer_order_history_df = country_order_join \
    .withColumn("Transaction_count", count(col("Order_ID")).over(window)) \
    .withColumn("Item_count", size(collect_set(col("Item")).over(window))) \
    .withColumn("Total_Amount_spent", _sum(col("Amount")).over(window)) \
    .withColumn("product_list", collect_set(col("Item")).over(window)) \
    .select("Customer_ID", "Transaction_count", "Item_count", "Total_Amount_spent", "product_list").show()

print("the maximum product purchased for each country")

total_Item_sale_by_country_df = country_order_join \
    .select("country", "Item") \
    .groupBy("Country", "Item") \
    .agg(count("Item").alias("Item_count_by_country"))

window = Window.partitionBy("country").orderBy("Item_count_by_country")

most_sale_item_by_country = total_Item_sale_by_country_df \
    .withColumn("rank_of_sale", rank().over(window)) \
    .filter(col("rank_of_sale") == 1).select("Country", "Item").show()

print("the most purchased product based on the age category less than 30 and above 30.")

purchased_item_by_age = country_order_join \
    .withColumn("age_range", when(col("Age") <= 30, lit("age_less_than_or_equalto__30")) \
                .otherwise(lit("age_grater_than_30"))) \
    .groupBy("age_range", "Item").agg(count("Item").alias("count"))

window = Window.partitionBy("age_range").orderBy("count")

most_purchased_item_by_age = purchased_item_by_age.withColumn("rank", rank().over(window)) \
    .filter(col("rank") == 1) \
    .select("age_range", "Item").show()

print("the country that had minimum transactions and sales amount.")

country_with_min_sale_and_transaction = country_order_join \
    .groupBy("Country") \
    .agg(count("Order_ID").alias("order_id_count"), _sum("Amount").alias("Amount_spent_by_country"))

min_order_country = country_with_min_sale_and_transaction.select("Country", "order_id_count") \
    .orderBy("order_id_count", ascending=True).take(1)[0]["Country"]

min_transaction_country = country_with_min_sale_and_transaction.select("Country", "order_id_count") \
    .orderBy("Amount_spent_by_country", ascending=True).take(1)[0]["Country"]

print("Country with min transaction:", min_order_country)
print("Country with min sales:", min_transaction_country)

country_order_join.unpersist()

spark.stop()
