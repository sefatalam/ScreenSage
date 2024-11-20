import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.Window

import scala.util.Random

object preparation{
  def main(args: Array[String]): Unit = {
    val conf = new SparkConf().setAppName("preparation").setMaster("local[*]")
    conf.set("spark.sql.analyzer.maxIterations", "5000")
    val sc = new SparkContext(conf)

    val spark = SparkSession
      .builder()
      .appName("preparation")
      .getOrCreate()
    import spark.implicits._

//    val titlebasics = args(2)
//    val titlecrew = args(3)
//    val titleepisode = args(4)
//    val titleprincipals = args(5)
//    val titleratings = args(6)
//    val input = args(7)
    val movies = args(0)
//    val testSet = args(1)

    val movies_df = spark.read.option("sep", "\t").option("header", value = true)
      .schema("tconst STRING, averageRating DOUBLE, numVotes INT, titleType STRING, originalTitle STRING, isAdult INT, runtimeMinutes INT, genres STRING, directors STRING, writers STRING")
      .csv(movies)

/** selecting relevant columns and joining them into one big Dataframe
 */

//    val titleratings_df = spark.read.option("sep", "\t").option("header", value = true)
//      .schema("tconst STRING, averageRating DOUBLE, numVotes INT")
//      .csv(titleratings)
//
//    val titlebasics_df = spark.read.option("sep", "\t").option("header", value = true)
//      .schema("tconst STRING, titleType STRING, primaryTitle STRING, originalTitle STRING, isAdult INT, startYear INT, endYear INT, runtimeMinutes INT, genres STRING")
//      .csv(titlebasics)
//      .drop("primaryTitle", "startYear", "endYear")
//
//
//    val titlecrew_df = spark.read.option("sep", "\t").option("header", value = true)
//      .schema("tconst STRING, directors STRING, writers STRING")
//      .csv(titlecrew)
//
//
//    val joined_df = titleratings_df.join(titlebasics_df, "tconst")
//      .join(titlecrew_df, "tconst")
//
//    val movies_df = joined_df.filter((joined_df("titleType") === "movie") || (joined_df("titleType") === "tvMovie") || (joined_df("titleType") === "tvSpecial"))


/** calculating the average runtimeMinutes, calculating the deviation of each movie's runtime from the average
*  adding binary columns to categorize movies based on their "titleType"
*/

    val average = movies_df.agg(avg($"runtimeMinutes")).first().getDouble(0)

    val average_df = movies_df
      .withColumn("avgRuntimeMinutes", $"runtimeMinutes" - average)
      .withColumn("is_movie", when($"titleType" === "movie", 1.0).otherwise(0.0))
      .withColumn("is_tvMovie", when($"titleType" === "tvMovie", 1.0).otherwise(0.0))
      .withColumn("is_tvSpecial", when($"titleType" === "tvSpecial", 1.0).otherwise(0.0))
      .select("tconst","avgRuntimeMinutes","is_movie","is_tvMovie","is_tvSpecial")

/** One-Hot encoding of top directors and genres
*/
    val sep_string = udf((str: String) => str.split(","))

    val genre_df = movies_df
      .withColumn("genres_arr", sep_string($"genres"))
      .drop("genres")
      .withColumnRenamed("genres_arr", "genres")

    val directors_df = movies_df
      .withColumn("directors_arr",sep_string($"directors"))
      .drop("directors")
      .withColumnRenamed("directors_arr", "directors")

    //extract top 3000 directors by number of movies directed
    val topDirectors = directors_df
      .select(explode(col("directors")) as "single_director")
      .select("single_director")
      .groupBy("single_director")
      .count()
      .sort(desc("count"))
      .collect()
      .map(_.getString(0))
      .slice(1,501)

    // Crate a UDF (User Defined Function), to check the presence of a specific genre and directors
    val containsGenreUDF = udf((genre: String, allGenres: Seq[String]) => allGenres.contains(genre))
    val containsDirectorsUDF = udf((director: String, allDirectors: Seq[String]) => allDirectors.contains(director))

    // Add a new column for each director
    val dfWithBinaryDirectors = topDirectors.foldLeft(directors_df) { (df, director) =>
      df.withColumn(s"${director}", when(containsDirectorsUDF(lit(director), col("directors")), 1.0).otherwise(0.0))
    }.drop("directors", "writers", "genres", "numVotes", "averageRating", "titleType", "originalTitle", "isAdult", "runtimeMinutes")

    // List all unique genres
    val distinctGenres = genre_df
      .select(explode(col("genres")) as "single_genre")
      .select("single_genre")
      .distinct()
      .collect()
      .map(_.getString(0))

    // Add a new column for each genre
    val dfWithBinaryGenres = distinctGenres.foldLeft(genre_df) { (df, genre) =>
      df.withColumn(s"is_${genre}", when(containsGenreUDF(lit(genre), col("genres")), 1.0).otherwise(0.0))}

    //Join both encoded dataframes
    val binaryMovies_df = dfWithBinaryGenres.join(average_df, "tconst").join(dfWithBinaryDirectors, "tconst")
      .drop("runtimeMinutes", "genres", "writers", "titleType", "directors")

    // binaryMovies_df.show()

/** Determine testSet & trainingSet
 */
    //Sort by numVotes and take every fourth movie to create testSet
    val ordering = Window.orderBy("numVotes") // order by "numVotes"
    val testSet = binaryMovies_df
      .withColumn("row_number", row_number().over(ordering))
      .filter($"row_number" % 4 === 0)
      .drop("row_number")
      .orderBy(rand()) // reset ordering

    val trainingSet = binaryMovies_df
      .withColumn("row_number", row_number().over(ordering))
      .filter($"row_number" % 4 =!= 0)
      .drop("row_number")
      .orderBy(rand()) // reset ordering

/** Write functions
  */
    binaryMovies_df.coalesce(1)
      .write
      .format("csv")
      .option("sep", "\t")
      .option("header", "true")
      .save(args(1))

    testSet.coalesce(1)
      .write
      .format("csv")
      .option("sep", "\t")
      .option("header", "true")
      .save(args(2))

    trainingSet.coalesce(1)
      .write
      .format("csv")
      .option("sep", "\t")
      .option("header", "true")
      .save(args(3))
  }
}

//.schema("tconst STRING, averageRating DOUBLE, numVotes DOUBLE, originalTitle STRING, isAdult DOUBLE, is_Crime DOUBLE, is_Romance " +
//        "DOUBLE, is_Thriller DOUBLE, is_Adventure DOUBLE, is_N DOUBLE, is_Drama DOUBLE, is_War DOUBLE, is_Documentary DOUBLE, is_Reality_TV DOUBLE," +
//        " is_Family DOUBLE, is_Fantasy DOUBLE, is_Game_Show DOUBLE, is_Adult DOUBLE, is_History DOUBLE, is_Mystery DOUBLE, is_Musical DOUBLE, " +
//        "is_Animation DOUBLE, is_Music DOUBLE, is_Film_Noir DOUBLE, is_Horror DOUBLE, is_Short DOUBLE, is_Western DOUBLE, is_Biography DOUBLE, " +
//        "is_Comedy DOUBLE, is_Action DOUBLE, is_Sport DOUBLE, is_Talk_Show DOUBLE, is_Sci_Fi DOUBLE, is_News DOUBLE, avgRuntimeMinutes DOUBLE, " +
//        "is_movie DOUBLE, is_tvMovie DOUBLE, is_tvSpecial DOUBLE, nm0242658 DOUBLE, nm0627864 DOUBLE, nm0001238 DOUBLE, nm0395776 DOUBLE, nm0002031 " +
//        "DOUBLE, nm0918999 DOUBLE, nm0064415 DOUBLE, nm0554924 DOUBLE, nm0782947 DOUBLE, nm0484645 DOUBLE, nm0488108 DOUBLE, nm0781261 DOUBLE, " +
//        "nm0861703 DOUBLE, nm0061792 DOUBLE, nm0357895 DOUBLE, nm0853028 DOUBLE, nm0000406 DOUBLE, nm0437356 DOUBLE, nm0213983 DOUBLE, nm0385171 " +
//        "DOUBLE, nm2269530 DOUBLE, nm0939147 DOUBLE, nm0792450 DOUBLE, nm0217035 DOUBLE, nm0947998 DOUBLE, nm0332985 DOUBLE, nm0676248 DOUBLE, " +
//        "nm0689245 DOUBLE, nm0071560 DOUBLE, nm0589495 DOUBLE, nm0567757 DOUBLE, nm0136579 DOUBLE, nm0136025 DOUBLE, nm0001090 DOUBLE, nm0245385 " +
//        "DOUBLE, nm0706484 DOUBLE, nm0768422 DOUBLE, nm0936823 DOUBLE, nm0005847 DOUBLE, nm0187671 DOUBLE, nm0172485 DOUBLE, nm0258581 DOUBLE, " +
//        "nm0347890 DOUBLE, nm0909825 DOUBLE, nm0654839 DOUBLE, nm0151653 DOUBLE, nm0218039 DOUBLE, nm0482774 DOUBLE, nm0945282 DOUBLE, nm0045800 " +
//        "DOUBLE, nm0218752 DOUBLE, nm0255386 DOUBLE, nm0201927 DOUBLE, nm0782682 DOUBLE, nm0589106 DOUBLE, nm0002815 DOUBLE, nm0655230 DOUBLE, " +
//        "nm0015570 DOUBLE, nm0765873 DOUBLE, nm0294758 DOUBLE, nm0006943 DOUBLE, nm0337586 DOUBLE, nm0127511 DOUBLE, nm0140241 DOUBLE, nm0376033 " +
//        "DOUBLE, nm0052897 DOUBLE, nm0920074 DOUBLE, nm0886754 DOUBLE, nm0631438 DOUBLE, nm0107854 DOUBLE, nm0851537 DOUBLE, nm0698184 DOUBLE, " +
//        "nm4341114 DOUBLE, nm0075318 DOUBLE, nm0565063 DOUBLE, nm0136552 DOUBLE, nm0002061 DOUBLE, nm0050048 DOUBLE, nm0226189 DOUBLE, nm0782707 " +
//        "DOUBLE, nm0197929 DOUBLE, nm0765430 DOUBLE, nm0769615 DOUBLE, nm0496505 DOUBLE, nm0030762 DOUBLE, nm0040220 DOUBLE, nm0002179 DOUBLE, " +
//        "nm0586281 DOUBLE, nm0826642 DOUBLE, nm0623401 DOUBLE, nm0946875 DOUBLE, nm0704841 DOUBLE, nm0550892 DOUBLE, nm0503777 DOUBLE, nm0819806 " +
//        "DOUBLE, nm0001124 DOUBLE, nm0920862 DOUBLE, nm0546791 DOUBLE, nm0406728 DOUBLE, nm0257638 DOUBLE, nm0135781 DOUBLE, nm2551464 DOUBLE, " +
//        "nm0691061 DOUBLE, nm0000419 DOUBLE, nm0210454 DOUBLE, nm0621540 DOUBLE, nm0408389 DOUBLE, nm0215877 DOUBLE, nm0609200 DOUBLE, nm0454771 " +
//        "DOUBLE, nm0006620 DOUBLE, nm0235066 DOUBLE, nm0641045 DOUBLE, nm0902006 DOUBLE, nm0502752 DOUBLE, nm0882190 DOUBLE, nm0419494 DOUBLE, " +
//        "nm0749914 DOUBLE, nm0334353 DOUBLE, nm0258015 DOUBLE, nm0128715 DOUBLE, nm0890060 DOUBLE, nm0102908 DOUBLE, nm0357143 DOUBLE, nm0560395 " +
//        "DOUBLE, nm0368871 DOUBLE, nm0317856 DOUBLE, nm0736610 DOUBLE, nm0812958 DOUBLE, nm0302778 DOUBLE, nm0946391 DOUBLE, nm0049335 DOUBLE, " +
//        "nm0393094 DOUBLE, nm0320946 DOUBLE, nm0430782 DOUBLE, nm0350947 DOUBLE, nm0721219 DOUBLE, nm0411030 DOUBLE, nm0408392 DOUBLE, nm0227762 " +
//        "DOUBLE, nm0282984 DOUBLE, nm0502391 DOUBLE, nm0059106 DOUBLE, nm0351411 DOUBLE, nm0297935 DOUBLE, nm0889402 DOUBLE, nm0001031 DOUBLE, " +
//        "nm0765121 DOUBLE, nm0497379 DOUBLE, nm0538632 DOUBLE, nm0523893 DOUBLE, nm0327950 DOUBLE, nm0002030 DOUBLE, nm0351645 DOUBLE, nm0124877 " +
//        "DOUBLE, nm0097648 DOUBLE, nm0292134 DOUBLE, nm0598102 DOUBLE, nm0523932 DOUBLE, nm0960375 DOUBLE, nm0939992 DOUBLE, nm0864775 DOUBLE, " +
//        "nm1991822 DOUBLE, nm0181530 DOUBLE, nm0879697 DOUBLE, nm0004630 DOUBLE, nm0245213 DOUBLE, nm0384616 DOUBLE, nm0015648 DOUBLE, nm0448915 " +
//        "DOUBLE, nm0004372 DOUBLE, nm0047971 DOUBLE, nm0000005 DOUBLE, nm0459567 DOUBLE, nm0735879 DOUBLE, nm0166836 DOUBLE, nm0723629 DOUBLE, " +
//        "nm0159201 DOUBLE, nm0615932 DOUBLE, nm0762517 DOUBLE, nm0592687 DOUBLE, nm0000465 DOUBLE, nm0509327 DOUBLE, nm0596410 DOUBLE, nm0006395 " +
//        "DOUBLE, nm0668201 DOUBLE, nm0066247 DOUBLE, nm0075520 DOUBLE, nm0833965 DOUBLE, nm0250625 DOUBLE, nm0023813 DOUBLE, nm0003474 DOUBLE, " +
//        "nm0728271 DOUBLE, nm0491504 DOUBLE, nm0412666 DOUBLE, nm0718243 DOUBLE, nm0627087 DOUBLE, nm0802563 DOUBLE, nm0409757 DOUBLE, nm0569645 " +
//        "DOUBLE, nm0179281 DOUBLE, nm0351531 DOUBLE, nm0547446 DOUBLE, nm0302143 DOUBLE, nm0912238 DOUBLE, nm0145336 DOUBLE, nm0000339 DOUBLE, " +
//        "nm0409036 DOUBLE, nm0289297 DOUBLE, nm0906709 DOUBLE, nm0477702 DOUBLE, nm0258902 DOUBLE, nm0557945 DOUBLE, nm0003422 DOUBLE, nm0068871 " +
//        "DOUBLE, nm0001348 DOUBLE, nm0485943 DOUBLE, nm6935209 DOUBLE, nm0160900 DOUBLE, nm0936204 DOUBLE, nm0030791 DOUBLE, nm0553941 DOUBLE, " +
//        "nm0603217 DOUBLE, nm0902823 DOUBLE, nm0176699 DOUBLE, nm0000033 DOUBLE, nm0351486 DOUBLE, nm0840671 DOUBLE, nm0000095 DOUBLE, nm0505610 " +
//        "DOUBLE, nm0546672 DOUBLE, nm0682749 DOUBLE, nm0500988 DOUBLE, nm0759232 DOUBLE, nm0175428 DOUBLE, nm0782597 DOUBLE, nm0878338 DOUBLE, " +
//        "nm1066739 DOUBLE, nm0329574 DOUBLE, nm0025926 DOUBLE, nm0560489 DOUBLE, nm0002086 DOUBLE, nm0303120 DOUBLE, nm0763798 DOUBLE, nm0515491 " +
//        "DOUBLE, nm0139878 DOUBLE, nm0160108 DOUBLE, nm0869665 DOUBLE, nm0561879 DOUBLE, nm0489660 DOUBLE, nm0229424 DOUBLE, nm0331964 DOUBLE, " +
//        "nm0247484 DOUBLE, nm0707952 DOUBLE, nm0152401 DOUBLE, nm0953130 DOUBLE, nm0166730 DOUBLE, nm0433657 DOUBLE, nm0896542 DOUBLE, nm0735416 " +
//        "DOUBLE, nm0943758 DOUBLE, nm0008152 DOUBLE, nm0004454 DOUBLE, nm0001486 DOUBLE, nm0523310 DOUBLE, nm0015037 DOUBLE, nm0495331 DOUBLE, " +
//        "nm0417871 DOUBLE, nm0439597 DOUBLE, nm2829542 DOUBLE, nm0064723 DOUBLE, nm0220131 DOUBLE, nm0279807 DOUBLE, nm0455839 DOUBLE, nm0379092 " +
//        "DOUBLE, nm0624756 DOUBLE, nm0157778 DOUBLE, nm0766211 DOUBLE, nm0086251 DOUBLE, nm0107664 DOUBLE, nm0434189 DOUBLE, nm0851253 DOUBLE, " +
//        "nm0620052 DOUBLE, nm0757157 DOUBLE, nm0845290 DOUBLE, nm0252751 DOUBLE, nm0310449 DOUBLE, nm0007139 DOUBLE, nm0346436 DOUBLE, nm0780764 " +
//        "DOUBLE, nm0003458 DOUBLE, nm1015063 DOUBLE, nm0491290 DOUBLE, nm0716340 DOUBLE, nm0000647 DOUBLE, nm0002089 DOUBLE, nm0616114 DOUBLE, " +
//        "nm0894204 DOUBLE, nm0880880 DOUBLE, nm0758508 DOUBLE, nm0414672 DOUBLE, nm0227661 DOUBLE, nm0191899 DOUBLE, nm0467396 DOUBLE, nm0515979 " +
//        "DOUBLE, nm0196614 DOUBLE, nm0179278 DOUBLE, nm0281808 DOUBLE, nm0126864 DOUBLE, nm0080315 DOUBLE, nm0386382 DOUBLE, nm0664515 DOUBLE, " +
//        "nm0003836 DOUBLE, nm0246473 DOUBLE, nm0827854 DOUBLE, nm0004466 DOUBLE, nm0338719 DOUBLE, nm0210322 DOUBLE, nm0281507 DOUBLE, nm0321159 " +
//        "DOUBLE, nm0113284 DOUBLE, nm0007179 DOUBLE, nm0001008 DOUBLE, nm0159741 DOUBLE, nm0654868 DOUBLE, nm0562845 DOUBLE, nm0268513 DOUBLE, " +
//        "nm0307819 DOUBLE, nm0781292 DOUBLE, nm0542720 DOUBLE, nm0613996 DOUBLE, nm0610303 DOUBLE, nm0188669 DOUBLE, nm0110653 DOUBLE, nm0429958 " +
//        "DOUBLE, nm3848412 DOUBLE, nm0205760 DOUBLE, nm0002194 DOUBLE, nm0890864 DOUBLE, nm0001328 DOUBLE, nm0496746 DOUBLE, nm0552871 DOUBLE, " +
//        "nm0000490 DOUBLE, nm0244421 DOUBLE, nm1328135 DOUBLE, nm0539034 DOUBLE, nm0548071 DOUBLE, nm3008388 DOUBLE, nm0002165 DOUBLE, nm0880618 " +
//        "DOUBLE, nm0286578 DOUBLE, nm0436382 DOUBLE, nm0408348 DOUBLE, nm0245081 DOUBLE, nm0652650 DOUBLE, nm0379391 DOUBLE, nm0227460 DOUBLE, " +
//        "nm0858873 DOUBLE, nm0454879 DOUBLE, nm0141448 DOUBLE, nm0646987 DOUBLE, nm0490487 DOUBLE, nm0906667 DOUBLE, nm0782804 DOUBLE, nm0936464 " +
//        "DOUBLE, nm0000485 DOUBLE, nm0076419 DOUBLE, nm0192698 DOUBLE, nm0975333 DOUBLE, nm0206560 DOUBLE, nm0928214 DOUBLE, nm0005866 DOUBLE, " +
//        "nm0840042 DOUBLE, nm0869645 DOUBLE, nm0921288 DOUBLE, nm0901138 DOUBLE, nm0862605 DOUBLE, nm0840651 DOUBLE, nm0557840 DOUBLE, nm0645661 " +
//        "DOUBLE, nm0864818 DOUBLE, nm0596850 DOUBLE, nm0921464 DOUBLE, nm0557927 DOUBLE, nm1048560 DOUBLE, nm0003226 DOUBLE, nm0661345 DOUBLE, " +
//        "nm0324875 DOUBLE, nm0767022 DOUBLE, nm0006837 DOUBLE, nm0765881 DOUBLE, nm0052630 DOUBLE, nm0124571 DOUBLE, nm0000776 DOUBLE, nm0000694 " +
//        "DOUBLE, nm0223522 DOUBLE, nm0572851 DOUBLE, nm1212040 DOUBLE, nm0435050 DOUBLE, nm0002175 DOUBLE, nm0793881 DOUBLE, nm0401680 DOUBLE, " +
//        "nm0213940 DOUBLE, nm0052677 DOUBLE, nm1390764 DOUBLE, nm0001379 DOUBLE, nm0251027 DOUBLE, nm0666685 DOUBLE, nm0000265 DOUBLE, nm1098369 " +
//        "DOUBLE, nm0335093 DOUBLE, nm0706350 DOUBLE, nm0593014 DOUBLE, nm0417054 DOUBLE, nm0882795 DOUBLE, nm0908831 DOUBLE, nm0023929 DOUBLE, " +
//        "nm0820461 DOUBLE, nm0739501 DOUBLE, nm0614634 DOUBLE, nm0956484 DOUBLE, nm0351410 DOUBLE, nm0256586 DOUBLE, nm0210811 DOUBLE, nm0389219 " +
//        "DOUBLE, nm0347728 DOUBLE, nm0904706 DOUBLE, nm0237424 DOUBLE, nm0555005 DOUBLE, nm0091380 DOUBLE, nm0836328 DOUBLE, nm0853928 DOUBLE, " +
//        "nm0823906 DOUBLE, nm0611531 DOUBLE, nm0035067 DOUBLE, nm0000217 DOUBLE, nm0829038 DOUBLE, nm2563700 DOUBLE, nm0162272 DOUBLE, nm0494110 " +
//        "DOUBLE, nm0416258 DOUBLE, nm0772191 DOUBLE, nm0190516 DOUBLE, nm0003941 DOUBLE, nm0587277 DOUBLE, nm0273477 DOUBLE, nm0863254 DOUBLE, " +
//        "nm0315117 DOUBLE, nm0006615 DOUBLE, nm0001175 DOUBLE, nm0630197 DOUBLE, nm0491626 DOUBLE, nm0500552 DOUBLE, nm0209853 DOUBLE, nm0814469 " +
//        "DOUBLE, nm0037571 DOUBLE, nm0618996 DOUBLE, nm0003373 DOUBLE, nm0295901 DOUBLE, nm0216381 DOUBLE, nm0542649 DOUBLE, nm0226244 DOUBLE, " +
//        "nm1714788 DOUBLE, nm0132414 DOUBLE, nm0308417 DOUBLE, nm0872062 DOUBLE, nm0539950 DOUBLE, nm0414985 DOUBLE, nm0105175 DOUBLE, nm0643171 " +
//        "DOUBLE, nm0868315 DOUBLE, nm0291548 DOUBLE, nm0879802 DOUBLE, nm0173728 DOUBLE, nm0928929 DOUBLE, nm0936404 DOUBLE, nm0103855 DOUBLE, " +
//        "nm0557857 DOUBLE, nm0759662 DOUBLE, nm0800547 DOUBLE, nm0899649 DOUBLE, nm0285831 DOUBLE, nm0847690 DOUBLE, nm0292720 DOUBLE, nm0689789 " +
//        "DOUBLE, nm0516360 DOUBLE, nm0213136 DOUBLE, nm0815638 DOUBLE, nm0001241 DOUBLE, nm0275269 DOUBLE, nm0744023 DOUBLE, nm0859387 DOUBLE, " +
//        "nm0072159 DOUBLE, nm0252475 DOUBLE, nm0903088 DOUBLE, nm0418130 DOUBLE, nm0862821 DOUBLE, nm0856915 DOUBLE, nm0050853 DOUBLE, nm0000142 " +
//        "DOUBLE")
