package org.example

import org.apache.hadoop.conf.Configuration
import org.apache.hadoop.fs.{FileSystem, FileUtil, Path}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.{DoubleType, LongType}





object App {

  def main (args: Array[String]): Unit ={

    val spark = org.apache.spark.sql.SparkSession.builder
      .master("local")
      .appName("Spark CSV Reader")
      .getOrCreate

    val df = spark.read.option("header",value = true).csv("src/main/resources/googleplaystore_user_reviews.csv")
      .select("App","Sentiment_Polarity")

    val df_1= df.withColumn("Sentiment_Polarity",col("Sentiment_Polarity").cast(DoubleType))
      .groupBy("App").avg("Sentiment_Polarity")
      .na.fill(0)
      .withColumnRenamed("avg(Sentiment_Polarity)","Average_Sentiment_Polarity")

    //df_1.printSchema()
    df_1.show
    //PART2

    val df_part2 = spark.read.option("header",value = true).csv("src/main/resources/googleplaystore.csv")
    df_part2.createOrReplaceTempView("data")
    val sqlDF_part2 = spark.sql("SELECT * FROM data WHERE RATING >= 4.0").orderBy(col("Rating").desc)
      .filter(!isnan(col("RATING"))) //Delete rows with NaN in column RATING

    sqlDF_part2.repartition(1).write.option("delimiter","§").csv("src/output/tmp/")
    sqlDF_part2.show


    val hadoopConfig = new Configuration()
    val hdfs = FileSystem.get(hadoopConfig)
    val srcPath = new Path("src/output/tmp")
    val destPath = new Path("src/output/best_apps.csv")
    FileUtil.copyMerge(hdfs, srcPath, hdfs, destPath, true, hadoopConfig, null)

    //remove ficheiro crc
    hdfs.delete(new Path("src/output/.best_apps.csv.crc"), true)


    //PART 3

    //sem duplicados [App,Category]
    val part3  = df_part2.select("App","Category").distinct()

    //concatenação das categorias
    val conc = part3.groupBy("App").agg(concat_ws(",", collect_list("Category")).alias("Categories"))
    val mid = df_part2.withColumnRenamed("App","App2")

    //junção com o resto do dataframe
    val compl = conc.join(mid,conc("App")===mid("App2")).dropDuplicates("App").drop("Category","App2")

    //renaming do  das colunas Last Updated, Content Rating, Current Ver, Android Ver
    //alteração do tipo de dados das colunas Rating, Reviews, Price
    val renam = compl.withColumn("Categories",split(col("Categories"),",").cast("array<string>"))
      .withColumn("Rating",col("Rating").cast(DoubleType))
      .withColumn("Reviews",col("Reviews").cast(LongType))
      //.withColumn("Price",col("Price").cast(DoubleType))
      .withColumnRenamed("Last Updated","Last_Updated")
      .withColumnRenamed("Content Rating","Content_Rating")
      .withColumnRenamed("Current Ver","Current_Version")
      .withColumnRenamed("Android Ver","Minimum_Android_Version")


    //parse da coluna Price em PriceINT
    val priceParse = renam.withColumn("PriceINT", regexp_extract(col("Price"),"(\\-?\\d*\\.?\\d+)",0)*0.9)
      .select("App","PriceINT")
      .withColumnRenamed("App","App2")

    //parse da coluna Size em SizeINT e SizeChar
    val sizeParse = renam.withColumn("SizeINT", regexp_extract(col("Size"),"(\\-?\\d*\\.?\\d+)",0))
      .withColumn("SizeCHAR",regexp_extract(col("Size"),"[a-z]+",0))
    //linhas com SizeCHAR == k => SizeINT/1000
    val rowsKb = sizeParse.filter(col("SizeCHAR").contains("k"))
      .withColumn("SizeINT",col("SizeINT")/1000)
    //linhas com Size em MB
    val rowsMb = sizeParse.filter(!col("SizeCHAR").contains("k"))

    val aux1 = rowsKb.select("App","SizeINT").withColumnRenamed("App","App2")
    val aux2 = rowsMb.select("App","SizeINT").withColumnRenamed("App","App2")

    //junção do preço eu euros, Kb e Mb ao dataframe
    val compl2 = renam.join(aux1,renam("App")===aux1("App2")).union(renam.join(aux2,renam("App")===aux2("App2")))
    val compl3 = compl2.join(priceParse,compl2("App")===priceParse("App2"))

    //formatação da ordem das colunas e alteração do nome SizeINT para Size e PriceINT para Price
    val data = compl3.select("App","Categories","Rating","Reviews","SizeINT","Installs","Type","PriceINT","Content_Rating","Genres","Last_Updated","Current_Version","Minimum_Android_Version")
      .withColumnRenamed("SizeINT","Size")
      .withColumnRenamed("PriceINT","Price")

    //alteração do tipo da coluna Genres
    val gen = data
      .withColumn("Genres",split(col("Genres"),";").cast("array<string>"))
    //conversão de String em DateType da coluna Last_Updated
    val df_03 = gen.withColumn("Last_Updated", to_date(col("Last_Updated"),"MMM dd, YYYY"))

    val df_3 = df_03.withColumn("Size",when(col("Size").equalTo(""),null).otherwise(col("Size")))
        .withColumn("Rating",when(col("Rating").equalTo("NaN"),null).otherwise(col("Rating")))

    df_3.show()
    df_3.printSchema()

    //PART4
    val aux3 = df_1.withColumnRenamed("App","App2")
    val df_04 = df_3.join(aux3,df_3("App")===aux3("App2")).drop("App2")
    val df_4 = df_04
        .withColumn("Average_Sentiment_Polarity", when(col("Average_Sentiment_Polarity").equalTo("0"),null)
        .otherwise(col("Average_Sentiment_Polarity")))

    df_4.write.option("compression","gzip").parquet("src/output/googleplaystore_cleaned")

    //PART5

    //explode_outer para nao contabilizar conjuntos de géneros como um género diferente
    val exp = df_3.withColumn("Genres",explode_outer(col("Genres")))

    val count = exp.groupBy("Genres").count()

    val rating = exp.groupBy("Genres").avg("Rating")
      .withColumnRenamed("Genres","Genres2")

    val aux4 = count.join(rating,count("Genres")===rating("Genres2")).drop("Genres2")
    val polarity = df_4.withColumn("Genres",explode_outer(col("Genres")))
      .groupBy("Genres").avg("Average_Sentiment_Polarity").withColumnRenamed("Genres","Genres3")

    val df_5 = aux4.join(polarity,aux4("Genres")===polarity("Genres3")).drop("Genres3")
      .withColumnRenamed("Genres","Genre")
      .withColumnRenamed("count","Count")
      .withColumnRenamed("avg(Rating)","Average_Rating")
      .withColumnRenamed("avg(Average_Sentiment_Polarity)","Average_Sentiment_Polarity")

    df_5.write.option("compression","gzip").parquet("src/output/googleplaystore_metrics")

  }

}
