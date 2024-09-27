from pyspark.sql import SparkSession, Window
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.types import (
    StructType,
    StructField,
    DoubleType,
    DecimalType,
    StringType,
    FloatType,
)
from datetime import datetime


spark = SparkSession.builder.getOrCreate()


evidences = (
    spark.read.parquet(
        "gs://open-targets-data-releases/24.06/output/etl/parquet/evidence"
    )
    .filter(
        F.col("datasourceId").isin(
            [
                "ot_genetics_portal",
                "gene_burden",
                "eva",
                "eva_somatic",
                "gene2phenotype",
                "orphanet",
                "cancer_gene_census",
                "intogen",
                "impc",
                "chembl",
            ]
        )
    )
    .persist()
)
ot_genetics = evidences.filter(F.col("datasourceId") == "ot_genetics_portal")

platform_v = "24.06"


def directionOfEffect(evidences, platform_v):
    """
    function to develop DoE assessment from OT evidences files, creating two columns:
    direction on target and direction on trait
    """

    # Create the paths using the version variable

    target_path = (
        f"gs://open-targets-data-releases/{platform_v}/output/etl/parquet/targets/"
    )
    mecact_path = f"gs://open-targets-data-releases/{platform_v}/output/etl/parquet/mechanismOfAction/"

    target = spark.read.parquet(target_path)
    mecact = spark.read.parquet(mecact_path)
    # 1# Make a list of variant of interest (Sequence ontology terms) to subset data of interest.
    ### Bear in mind that SO works with ontology structure as: SO:XXXXXX, but databases has the SO as: SO_XXXXXX

    var_filter_lof = [
        ### High impact variants https://www.ensembl.org/info/genome/variation/prediction/predicted_data.html
        "SO_0001589",  ## frameshit_variant
        "SO_0001587",  ## stop_gained
        "SO_0001574",  ## splice_acceptor_variant
        "SO_0001575",  ## splice_donor_variant
        "SO_0002012",  ## start_lost
        "SO_0001578",  ## stop_lost
        "SO_0001893",  ## transcript_ablation
        # "SO:0001889", ## transcript_amplification ## the Only HIGH impact that increase protein.
    ]

    gof = ["SO_0002053"]
    lof = ["SO_0002054"]

    ## annotate TSG/oncogene/bivalent using 'hallmarks.attributes'
    oncotsg_list = [
        "TSG",
        "oncogene",
        "Oncogene",
        "oncogene",
        "oncogene,TSG",
        "TSG,oncogene",
        "fusion,oncogene",
        "oncogene,fusion",
    ]

    inhibitors = [
        "RNAI INHIBITOR",
        "NEGATIVE MODULATOR",
        "NEGATIVE ALLOSTERIC MODULATOR",
        "ANTAGONIST",
        "ANTISENSE INHIBITOR",
        "BLOCKER",
        "INHIBITOR",
        "DEGRADER",
        "INVERSE AGONIST",
        "ALLOSTERIC ANTAGONIST",
        "DISRUPTING AGENT",
    ]

    activators = [
        "PARTIAL AGONIST",
        "ACTIVATOR",
        "POSITIVE ALLOSTERIC MODULATOR",
        "POSITIVE MODULATOR",
        "AGONIST",
        "SEQUESTERING AGENT",
        "STABILISER",
    ]

    actionType = (
        mecact.select(
            F.explode_outer("chemblIds").alias("drugId2"),
            "actionType",
            "mechanismOfAction",
            "targets",
        )
        .select(
            F.explode_outer("targets").alias("targetId2"),
            "drugId2",
            "actionType",
            "mechanismOfAction",
        )
        .groupBy("targetId2", "drugId2")
        .agg(
            F.collect_set("actionType").alias("actionType"),
        )
    )

    oncolabel = (
        target.select(
            "id", "approvedSymbol", F.explode_outer(F.col("hallmarks.attributes"))
        )
        .select("id", "approvedSymbol", "col.description")
        .filter(F.col("description").isin(oncotsg_list))
        .groupBy("id", "approvedSymbol")
        .agg(F.collect_set("description").alias("description"))
        .withColumn("description_splited", F.concat_ws(",", F.col("description")))
        .withColumn(
            "TSorOncogene",
            F.when(
                (
                    F.col("description_splited").rlike("ncogene")
                    & F.col("description_splited").rlike("TSG")
                ),
                F.lit("bivalent"),
            )
            .when(
                F.col("description_splited").rlike("ncogene(\s|$)"), F.lit("oncogene")
            )
            .when(F.col("description_splited").rlike("TSG(\s|$)"), F.lit("TSG"))
            .otherwise(F.lit("noEvaluable")),
        )
        .withColumnRenamed("id", "target_id")
    )

    # 2# run the transformation of the evidences datasets used.
    all = evidences.filter(
        F.col("datasourceId").isin(
            [
                "ot_genetics_portal",
                "gene_burden",
                "eva",
                "eva_somatic",
                "gene2phenotype",
                "orphanet",
                "cancer_gene_census",
                "intogen",
                "impc",
                "chembl",
            ]
        )
    )

    windowSpec = Window.partitionBy("targetId", "diseaseId")

    assessment = (
        all.withColumn(
            "beta", F.col("beta").cast("double")
        )  ## ot genetics & gene burden
        .withColumn(
            "OddsRatio", F.col("OddsRatio").cast("double")
        )  ## ot genetics & gene burden
        .withColumn(
            "clinicalSignificances", F.concat_ws(",", F.col("clinicalSignificances"))
        )  ### eva
        .join(oncolabel, oncolabel.target_id == F.col("targetId"), "left")  ###  cgc
        .join(
            actionType,  ## chembl
            (actionType.drugId2 == F.col("drugId"))
            & (actionType.targetId2 == F.col("targetId")),
            "left",
        )
        .withColumn("inhibitors_list", F.array([F.lit(i) for i in inhibitors]))
        .withColumn("activators_list", F.array([F.lit(i) for i in activators]))
        .withColumn(
            "intogen_function",
            F.when(
                F.arrays_overlap(
                    F.col("mutatedSamples.functionalConsequenceId"),
                    F.array([F.lit(i) for i in (gof)]),
                ),
                F.lit("GoF"),
            ).when(
                F.arrays_overlap(
                    F.col("mutatedSamples.functionalConsequenceId"),
                    F.array([F.lit(i) for i in (lof)]),
                ),
                F.lit("LoF"),
            ),
            # .otherwise("nodata"),
        )
        .withColumn(
            "intogenAnnot",
            F.size(F.collect_set(F.col("intogen_function")).over(windowSpec)),
        )
        ### variant Effect Column
        .withColumn(
            "variantEffect",
            F.when(
                F.col("datasourceId") == "ot_genetics_portal",
                F.when(
                    F.col("variantFunctionalConsequenceId").isNotNull(),
                    F.when(
                        F.col("variantFunctionalConsequenceFromQtlId").isNull(),
                        F.when(
                            F.col("variantFunctionalConsequenceId").isin(
                                var_filter_lof
                            ),
                            F.lit("LoF"),
                        )
                        .when(
                            F.col("variantFunctionalConsequenceId").isin(gof),
                            F.lit("GoF"),
                        )
                        .otherwise(F.lit("noEvaluable")),
                    )
                    ### variantFunctionalConsequenceFromQtlId
                    .when(
                        F.col("variantFunctionalConsequenceFromQtlId").isNotNull(),
                        F.when(
                            F.col("variantFunctionalConsequenceId").isin(
                                var_filter_lof
                            ),  ## when is a LoF variant
                            F.when(
                                F.col("variantFunctionalConsequenceFromQtlId")
                                == "SO_0002316",
                                F.lit("LoF"),
                            )
                            .when(
                                F.col("variantFunctionalConsequenceFromQtlId")
                                == "SO_0002315",
                                F.lit("conflict/noEvaluable"),
                            )
                            .otherwise(F.lit("LoF")),
                        ).when(
                            F.col("variantFunctionalConsequenceId").isin(var_filter_lof)
                            == False,  ## when is not a LoF, still can be a GoF
                            F.when(
                                F.col("variantFunctionalConsequenceId").isin(gof)
                                == False,  ##if not GoF
                                F.when(
                                    F.col("variantFunctionalConsequenceFromQtlId")
                                    == "SO_0002316",
                                    F.lit("LoF"),
                                )
                                .when(
                                    F.col("variantFunctionalConsequenceFromQtlId")
                                    == "SO_0002315",
                                    F.lit("GoF"),
                                )
                                .otherwise(F.lit("noEvaluable")),
                            ).when(
                                F.col("variantFunctionalConsequenceId").isin(
                                    gof
                                ),  ##if is GoF
                                F.when(
                                    F.col("variantFunctionalConsequenceFromQtlId")
                                    == "SO_0002316",
                                    F.lit("conflict/noEvaluable"),
                                ).when(
                                    F.col("variantFunctionalConsequenceFromQtlId")
                                    == "SO_0002315",
                                    F.lit("GoF"),
                                ),
                            ),
                        ),
                    ),
                ).when(
                    F.col("variantFunctionalConsequenceId").isNull(),
                    F.when(
                        F.col("variantFunctionalConsequenceFromQtlId") == "SO_0002316",
                        F.lit("LoF"),
                    )
                    .when(
                        F.col("variantFunctionalConsequenceFromQtlId") == "SO_0002315",
                        F.lit("GoF"),
                    )
                    .otherwise(F.lit("noEvaluable")),
                ),
            ).when(
                F.col("datasourceId") == "gene_burden",
                F.when(F.col("targetId").isNotNull(), F.lit("LoF")).otherwise(
                    F.lit("noEvaluable")
                ),
            )
            #### Eva_germline
            .when(
                F.col("datasourceId") == "eva",
                F.when(
                    F.col("variantFunctionalConsequenceId").isin(var_filter_lof),
                    F.lit("LoF"),
                ).otherwise(F.lit("noEvaluable")),
            )
            #### Eva_somatic
            .when(
                F.col("datasourceId") == "eva_somatic",
                F.when(
                    F.col("variantFunctionalConsequenceId").isin(var_filter_lof),
                    F.lit("LoF"),
                ).otherwise(F.lit("noEvaluable")),
            )
            #### G2P
            .when(
                F.col("datasourceId")
                == "gene2phenotype",  ### 6 types of variants [SO_0002318, SO_0002317, SO_0001622, SO_0002315, SO_0001566, SO_0002220]
                F.when(
                    F.col("variantFunctionalConsequenceId") == "SO_0002317",
                    F.lit("LoF"),
                )  ### absent gene product
                .when(
                    F.col("variantFunctionalConsequenceId") == "SO_0002315",
                    F.lit("GoF"),
                )  ### increased gene product level
                .otherwise(F.lit("noEvaluable")),
            )
            #### Orphanet
            .when(
                F.col("datasourceId") == "orphanet",
                F.when(
                    F.col("variantFunctionalConsequenceId") == "SO_0002054",
                    F.lit("LoF"),
                )  ### Loss of Function Variant
                .when(
                    F.col("variantFunctionalConsequenceId") == "SO_0002053",
                    F.lit("GoF"),
                )  ### Gain_of_Function Variant
                .otherwise(F.lit("noEvaluable")),
            )
            #### CGC
            .when(
                F.col("datasourceId") == "cancer_gene_census",
                F.when(F.col("TSorOncogene") == "oncogene", F.lit("GoF"))
                .when(F.col("TSorOncogene") == "TSG", F.lit("LoF"))
                .when(F.col("TSorOncogene") == "bivalent", F.lit("bivalent"))
                .otherwise("noEvaluable"),
            )
            #### intogen
            .when(
                F.col("datasourceId") == "intogen",
                F.when(
                    F.col("intogenAnnot")
                    == 1,  ## oncogene/tummor suppressor for a given trait
                    F.when(
                        F.arrays_overlap(
                            F.col("mutatedSamples.functionalConsequenceId"),
                            F.array([F.lit(i) for i in (gof)]),
                        ),
                        F.lit("GoF"),
                    ).when(
                        F.arrays_overlap(
                            F.col("mutatedSamples.functionalConsequenceId"),
                            F.array([F.lit(i) for i in (lof)]),
                        ),
                        F.lit("LoF"),
                    ),
                )
                .when(
                    F.col("intogenAnnot") > 1, F.lit("bivalentIntogen")
                )  ##oncogene & tumor suppressor for a given trait
                .otherwise(F.lit("noEvaluable")),
            )
            #### impc
            .when(
                F.col("datasourceId") == "impc",
                F.when(F.col("diseaseId").isNotNull(), F.lit("LoF")).otherwise(
                    F.lit("noEvaluable")
                ),
            )
            ### chembl
            .when(
                F.col("datasourceId") == "chembl",
                F.when(
                    F.size(
                        F.array_intersect(F.col("actionType"), F.col("inhibitors_list"))
                    )
                    >= 1,
                    F.lit("LoF"),
                )
                .when(
                    F.size(
                        F.array_intersect(F.col("actionType"), F.col("activators_list"))
                    )
                    >= 1,
                    F.lit("GoF"),
                )
                .otherwise(F.lit("noEvaluable")),
            ),
        )
        .withColumn(
            "directionOnTrait",
            ## ot genetics portal
            F.when(
                F.col("datasourceId") == "ot_genetics_portal",
                F.when(
                    (F.col("beta").isNotNull()) & (F.col("OddsRatio").isNull()),
                    F.when(F.col("beta") > 0, F.lit("risk"))
                    .when(F.col("beta") < 0, F.lit("protect"))
                    .otherwise(F.lit("noEvaluable")),
                )
                .when(
                    (F.col("beta").isNull()) & (F.col("OddsRatio").isNotNull()),
                    F.when(F.col("OddsRatio") > 1, F.lit("risk"))
                    .when(F.col("OddsRatio") < 1, F.lit("protect"))
                    .otherwise(F.lit("noEvaluable")),
                )
                .when(
                    (F.col("beta").isNull()) & (F.col("OddsRatio").isNull()),
                    F.lit("noEvaluable"),
                )
                .when(
                    (F.col("beta").isNotNull()) & (F.col("OddsRatio").isNotNull()),
                    F.lit("conflict/noEvaluable"),
                ),
            ).when(
                F.col("datasourceId") == "gene_burden",
                F.when(
                    (F.col("beta").isNotNull()) & (F.col("OddsRatio").isNull()),
                    F.when(F.col("beta") > 0, F.lit("risk"))
                    .when(F.col("beta") < 0, F.lit("protect"))
                    .otherwise(F.lit("noEvaluable")),
                )
                .when(
                    (F.col("oddsRatio").isNotNull()) & (F.col("beta").isNull()),
                    F.when(F.col("oddsRatio") > 1, F.lit("risk"))
                    .when(F.col("oddsRatio") < 1, F.lit("protect"))
                    .otherwise(F.lit("noEvaluable")),
                )
                .when(
                    (F.col("beta").isNull()) & (F.col("oddsRatio").isNull()),
                    F.lit("noEvaluable"),
                )
                .when(
                    (F.col("beta").isNotNull()) & (F.col("oddsRatio").isNotNull()),
                    F.lit("conflict"),
                ),
            )
            ## Eva_germline
            .when(
                F.col("datasourceId") == "eva",
                F.when(
                    F.col("clinicalSignificances").rlike("(pathogenic)$"), F.lit("risk")
                )
                .when(
                    F.col("clinicalSignificances").contains("protect"), F.lit("protect")
                )
                .otherwise(F.lit("noEvaluable")),
            )
            #### Eva_somatic
            .when(
                F.col("datasourceId") == "eva_somatic",
                F.when(
                    F.col("clinicalSignificances").rlike("(pathogenic)$"), F.lit("risk")
                )
                .when(
                    F.col("clinicalSignificances").contains("protect"), F.lit("protect")
                )
                .otherwise(F.lit("noEvaluable")),
            )
            #### G2P
            .when(
                F.col("datasourceId") == "gene2phenotype",
                F.when(F.col("diseaseId").isNotNull(), F.lit("risk")).otherwise(
                    F.lit("noEvaluable")
                ),
            )
            #### Orphanet
            .when(
                F.col("datasourceId") == "orphanet",
                F.when(F.col("diseaseId").isNotNull(), F.lit("risk")).otherwise(
                    F.lit("noEvaluable")
                ),
            )
            #### CGC
            .when(
                F.col("datasourceId") == "cancer_gene_census",
                F.when(F.col("diseaseId").isNotNull(), F.lit("risk")).otherwise(
                    F.lit("noEvaluable")
                ),
            )
            #### intogen
            .when(
                F.col("datasourceId") == "intogen",
                F.when(F.col("diseaseId").isNotNull(), F.lit("risk")).otherwise(
                    F.lit("noEvaluable")
                ),
            )
            #### impc
            .when(
                F.col("datasourceId") == "impc",
                F.when(F.col("diseaseId").isNotNull(), F.lit("risk")).otherwise(
                    F.lit("noEvaluable")
                ),
            )
            ### chembl
            .when(
                F.col("datasourceId") == "chembl",
                F.when(F.col("diseaseId").isNotNull(), F.lit("protect")).otherwise(
                    F.lit("noEvaluable")
                ),
            ),
        )
        .withColumn(
            "homogenized",
            F.when(
                (F.col("variantEffect") == "LoF")
                & (F.col("directionOnTrait") == "risk"),
                F.lit("LoF_risk"),
            )
            .when(
                (F.col("variantEffect") == "LoF")
                & (F.col("directionOnTrait") == "protect"),
                F.lit("LoF_protect"),
            )
            .when(
                (F.col("variantEffect") == "GoF")
                & (F.col("directionOnTrait") == "risk"),
                F.lit("GoF_risk"),
            )
            .when(
                (F.col("variantEffect") == "GoF")
                & (F.col("directionOnTrait") == "protect"),
                F.lit("GoF_protect"),
            )
            .otherwise(F.lit("noEvaluable")),
        )
    ).persist()

    return assessment


def discrepancifier(df):
    """
    detect discrepancies per row where there are the four
    DoE assessments using Null and isNotNull assessments
    """
    columns = ["GoF_risk", "LoF_protect", "LoF_risk", "GoF_protect", "noEvaluable"]

    for col in columns:
        if col not in df.columns:
            df = df.withColumn(col, F.lit(None)).persist()

    return df.withColumn(
        "coherencyDiagonal",
        F.when(
            (F.col("LoF_risk").isNull())
            & (F.col("LoF_protect").isNull())
            & (F.col("GoF_risk").isNull())
            & (F.col("GoF_protect").isNull())
            & (F.col("noEvaluable").isNull()),
            F.lit("noEvid"),
        )
        .when(
            (F.col("LoF_risk").isNull())
            & (F.col("LoF_protect").isNull())
            & (F.col("GoF_risk").isNull())
            & (F.col("GoF_protect").isNull())
            & (F.col("noEvaluable").isNotNull()),
            F.lit("EvidNotDoE"),
        )
        .when(
            (F.col("LoF_risk").isNotNull())
            | (F.col("LoF_protect").isNotNull())
            | (F.col("GoF_risk").isNotNull())
            | (F.col("GoF_protect").isNotNull()),
            F.when(
                ((F.col("GoF_risk").isNotNull()) & (F.col("LoF_risk").isNotNull())),
                F.lit("dispar"),
            )
            .when(
                ((F.col("LoF_protect").isNotNull()) & (F.col("LoF_risk").isNotNull())),
                F.lit("dispar"),
            )
            .when(
                ((F.col("GoF_protect").isNotNull()) & (F.col("GoF_risk").isNotNull())),
                F.lit("dispar"),
            )
            .when(
                (
                    (F.col("GoF_protect").isNotNull())
                    & (F.col("LoF_protect").isNotNull())
                ),
                F.lit("dispar"),
            )
            .otherwise(F.lit("coherent")),
        ),
    ).withColumn(
        "coherencyOneCell",
        F.when(
            (F.col("LoF_risk").isNull())
            & (F.col("LoF_protect").isNull())
            & (F.col("GoF_risk").isNull())
            & (F.col("GoF_protect").isNull())
            & (F.col("noEvaluable").isNull()),
            F.lit("noEvid"),
        )
        .when(
            (F.col("LoF_risk").isNull())
            & (F.col("LoF_protect").isNull())
            & (F.col("GoF_risk").isNull())
            & (F.col("GoF_protect").isNull())
            & (F.col("noEvaluable").isNotNull()),
            F.lit("EvidNotDoE"),
        )
        .when(
            (F.col("LoF_risk").isNotNull())
            | (F.col("LoF_protect").isNotNull())
            | (F.col("GoF_risk").isNotNull())
            | (F.col("GoF_protect").isNotNull()),
            F.when(
                F.col("LoF_risk").isNotNull()
                & (
                    (F.col("LoF_protect").isNull())
                    & (F.col("GoF_risk").isNull())
                    & (F.col("GoF_protect").isNull())
                ),
                F.lit("coherent"),
            )
            .when(
                F.col("GoF_risk").isNotNull()
                & (
                    (F.col("LoF_protect").isNull())
                    & (F.col("LoF_risk").isNull())
                    & (F.col("GoF_protect").isNull())
                ),
                F.lit("coherent"),
            )
            .when(
                F.col("LoF_protect").isNotNull()
                & (
                    (F.col("LoF_risk").isNull())
                    & (F.col("GoF_risk").isNull())
                    & (F.col("GoF_protect").isNull())
                ),
                F.lit("coherent"),
            )
            .when(
                F.col("GoF_protect").isNotNull()
                & (
                    (F.col("LoF_protect").isNull())
                    & (F.col("GoF_risk").isNull())
                    & (F.col("LoF_risk").isNull())
                ),
                F.lit("coherent"),
            )
            .otherwise(F.lit("dispar")),
        ),
    )