ThisBuild / version := "0.1.0-SNAPSHOT"
ThisBuild / scalaVersion := "2.12.17"

lazy val root = (project in file("."))

libraryDependencies ++= Seq(
    "org.apache.spark" %% "spark-core" % "3.3.1",
    "org.apache.spark" %% "spark-sql"  % "3.3.1"
)
