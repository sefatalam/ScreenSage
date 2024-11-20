import org.apache.commons.math3.util.FastMath.{abs, max}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.{SparkConf, SparkContext}
import org.json4s.scalap.scalasig.ClassFileParser.header
import org.apache.spark.sql.types._
import org.apache.spark.sql.functions._


case class MovieInternal(tconst: String,
                         averageRating: Double,
                         numVotes: Int,
                         titleType: String,
                         originalTitle: String,
                         isAdult: Int,
                         runtimeMinutes: String,
                         genres: Array[String],
                         directors: Array[String],
                         writers: Array[String],
                         similarity: Double)

case class MovieInput(averageRating: String,
                      titleType: String,
                      originalTitle: String,
                      isAdult: Int,
                      runtimeMinutes: String,
                      genres: Array[String],
                      directors: Array[String],
                      writers: Array[String])


object Joinimdb {
    def stringToInt(str: String) : Int = {
        if (str == "\\N" || str == null) {
            return 0
        }
        str.toInt
    }

    def arraySimilarity(arr1: Array[String], arr2: Array[String]) : Double = {
        if (arr1.contains("\\N") || arr2.contains("\\N") || arr1 == null || arr2 == null) {
            return 0.0
        }
        val intersect = arr1.intersect(arr2)

        intersect.length / max(arr1.length, arr2.length)
    }

    def isAdultSimilarity(int1: Int, int2: Int) : Double = {
        if (int1 == int2) {
            return 1.0
        }
        0.0
    }

    def runtimeSimilarity(str1: String, str2: String) : Double = {
        val int1 = stringToInt(str1)
        val int2 = stringToInt(str2)
        1.0 - abs(int1 - int2) / (int1 + int2 + 0.01)
    }

    def main(args: Array[String]) : Unit = {
        val conf = new SparkConf().setAppName("Joinimdb").setMaster("local[*]")
        val sc = new SparkContext(conf)

        val spark = SparkSession
          .builder()
          .appName("Joinimdb")
          .getOrCreate()
        import spark.implicits._

        if (args.length < 9) {
            println(">>> [ERROR] Expected arguments: namebasics file, titleakas file, titlebasics file, titlecrew file, titleepisodes file, titleprinciples file, titleratings file, input file, output path")
            return
        }

        // val namebasics = args(0)
        // val titleakas = args(1)
        val titlebasics = args(2)
        val titlecrew = args(3)
        val titleepisode = args(4)
        val titleprincipals = args(5)
        val titleratings = args(6)
        val input = args(7)


        val isAdult_weight = 1.0
        val runtime_weight = 1.0
        val director_weight = 2.0
        val writer_weight = 2.0
        val genre_weight = 1.2

        val complete_weight = isAdult_weight + runtime_weight + director_weight + writer_weight + genre_weight


        // Custom function to split string into array of string
        val sep_string = udf((str: String) => str.split(","))


        val titlebasics_df = spark.read.option("sep", "\t").option("header",value=true)
          .schema("tconst STRING, titleType STRING, primaryTitle STRING, originalTitle STRING, isAdult INT, startYear INT, endYear INT, runtimeMinutes INT, genres STRING")
          .csv(titlebasics).drop("primaryTitle", "startYear", "endYear")
          .withColumn("genres_arr", sep_string($"genres"))
          .drop("genres")
          .withColumnRenamed("genres_arr", "genres")

        val titlecrew_df = spark.read.option("sep", "\t").option("header",value=true)
          .schema("tconst STRING, directors STRING, writers STRING")
          .csv(titlecrew)
          .withColumn("directors_arr", sep_string($"directors"))
          .drop("directors")
          .withColumnRenamed("directors_arr", "directors")
          .withColumn("writers_arr", sep_string($"writers"))
          .drop("writers")
          .withColumnRenamed("writers_arr", "writers")

        //val titleepisode_df = spark.read.option("sep", "\t").option("header",value=true)
        // .schema("tconst STRING, parentTconst STRING, seasonNumber INT, episodeNumber INT")
        //  .csv(titleepisode)
        //  .drop("seasonNumber", "episodeNumber")

        //val titleprincipals_df = spark.read.option("sep", "\t").option("header",value=true)
        // .schema("tconst STRING, ordering INT, nconst STRING, category STRING, job STRING, characters STRING")
        // .csv(titleprincipals)

        val titleratings_df = spark.read.option("sep", "\t").option("header",value=true)
          .schema("tconst STRING, averageRating DOUBLE, numVotes INT")
          .csv(titleratings)


        // Input: one complete row: titleType   originalTitle   isAdult runtimeMinutes  genres  directors   writers
        // tconst, averageRating, numVotes all '\N'
        val input_rows = spark.read.option("sep", "\t")
          .schema("averageRating STRING, titleType STRING, originalTitle STRING, isAdult INT, runtimeMinutes INT, genres STRING, directors STRING, writers STRING")
          .csv(input)
          .withColumn("genres_arr", sep_string($"genres"))
          .drop("genres")
          .withColumnRenamed("genres_arr", "genres")
          .withColumn("directors_arr", sep_string($"directors"))
          .drop("directors")
          .withColumnRenamed("directors_arr", "directors")
          .withColumn("writers_arr", sep_string($"writers"))
          .drop("writers")
          .withColumnRenamed("writers_arr", "writers")
          .as[MovieInput].collect()

        val joined_df = titleratings_df.join(titlebasics_df, "tconst")
          .join(titlecrew_df, "tconst")


        // Every row one by one
        var predicted_ratings = new Array[String](input_rows.length)
        var actual_ratings = new Array[String](input_rows.length)
        var index = 0
        for (input_movie <- input_rows) {
            // Solely focussing on movies for now, rest later
            val movies_df = joined_df.filter(joined_df("titleType") === "movie").withColumn("similarity", lit(0.0)).as[MovieInternal]

            // Calculating similarities for movies that are NOT the input
            val similar_movies_df = movies_df.filter(movies_df("originalTitle") =!= input_movie.originalTitle).map(row => MovieInternal(row.tconst: String,
                row.averageRating,
                row.numVotes,
                row.titleType,
                row.originalTitle,
                row.isAdult,
                row.runtimeMinutes,
                row.genres,
                row.directors,
                row.writers,
                (row.similarity
                  + isAdultSimilarity(row.isAdult, input_movie.isAdult) * isAdult_weight
                  + runtimeSimilarity(row.runtimeMinutes, input_movie.runtimeMinutes) * runtime_weight
                  + arraySimilarity(row.genres, input_movie.genres) * genre_weight
                  + arraySimilarity(row.writers, input_movie.writers) * writer_weight
                  + arraySimilarity(row.directors, input_movie.directors) * director_weight
                  ) / complete_weight))

            val sim_movie_array = similar_movies_df
              .sort(similar_movies_df("similarity").desc)
              .limit(50).collect()

            var total_rating = 0.0
            var total_weights = 0.0
            for (movie <- sim_movie_array) {
                total_rating += movie.averageRating * movie.numVotes * movie.similarity
                total_weights += movie.numVotes * movie.similarity
            }

            predicted_ratings(index) = (total_rating / total_weights).toString.take(3)
            actual_ratings(index) = input_movie.averageRating
            index += 1
        }

        println("===========================================================================")
        println("RATING IMDB     ", actual_ratings.mkString(" Array(", ", ", ")"))
        println("RATING PREDICTED", predicted_ratings.mkString(" Array(", ", ", ")"))
        println("===========================================================================")


        // Convert arrays to comma-separated strings
        val joined_df_csv = joined_df
          .withColumn("genres_new", concat_ws(",", $"genres"))
          .drop("genres")
          .withColumn("directors_new", concat_ws(",", $"directors"))
          .drop("directors")
          .withColumn("writers_new", concat_ws(",", $"writers"))
          .drop("writers")

        joined_df_csv.coalesce(1)
          .write
          .format("csv")
          .option("sep", "\t")
          .option("header", "true")
          .save(args(8))
    }
}
