package edu.umd.cs.aspect

import org.slf4j.Logger
import org.slf4j.LoggerFactory
import com.google.common.collect.Iterables
import util.DataOutputter;
import util.FoldUtils;
import util.GroundingWrapper;
import edu.umd.cs.psl.application.inference.LazyMPEInference;
import edu.umd.cs.psl.application.inference.MPEInference
import edu.umd.cs.psl.application.learning.weight.em.*;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.LazyMaxLikelihoodMPE;
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxLikelihoodMPE
import edu.umd.cs.psl.application.learning.weight.maxlikelihood.MaxPseudoLikelihood
import edu.umd.cs.psl.application.learning.weight.maxmargin.MaxMargin
import edu.umd.cs.psl.config.*
import edu.umd.cs.psl.database.DataStore
import edu.umd.cs.psl.database.Database;
import edu.umd.cs.psl.database.DatabasePopulator
import edu.umd.cs.psl.database.DatabaseQuery
import edu.umd.cs.psl.database.Partition;
import edu.umd.cs.psl.database.ReadOnlyDatabase;
import edu.umd.cs.psl.database.ResultList
import edu.umd.cs.psl.database.rdbms.RDBMSDataStore
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver
import edu.umd.cs.psl.database.rdbms.driver.H2DatabaseDriver.Type
import edu.umd.cs.psl.evaluation.result.*
import edu.umd.cs.psl.evaluation.statistics.*
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics
import edu.umd.cs.psl.evaluation.statistics.RankingScore
import edu.umd.cs.psl.evaluation.statistics.SimpleRankingComparator
import edu.umd.cs.psl.evaluation.statistics.DiscretePredictionStatistics.BinaryClass;
import edu.umd.cs.psl.evaluation.statistics.filter.MaxValueFilter
import edu.umd.cs.psl.groovy.PSLModel;
import edu.umd.cs.psl.groovy.PredicateConstraint;
import edu.umd.cs.psl.groovy.SetComparison;
import edu.umd.cs.psl.model.Model;
import edu.umd.cs.psl.model.argument.ArgumentType;
import edu.umd.cs.psl.model.argument.GroundTerm;
import edu.umd.cs.psl.model.argument.Variable
import edu.umd.cs.psl.model.atom.GroundAtom;
import edu.umd.cs.psl.model.atom.QueryAtom
import edu.umd.cs.psl.model.atom.RandomVariableAtom
import edu.umd.cs.psl.model.function.ExternalFunction;
import edu.umd.cs.psl.ui.functions.textsimilarity.*
import edu.umd.cs.psl.ui.loading.InserterUtils;
import edu.umd.cs.psl.util.database.Queries;
import edu.umd.cs.psl.model.kernel.CompatibilityKernel
import edu.umd.cs.psl.model.parameters.Weight
import edu.umd.cs.psl.model.predicate.Predicate;
import edu.umd.cs.psl.model.kernel.predicateconstraint.*


class UnsupervisedFineThread{
	public static void main(String[] args)
	{
		for (int d = 1;d<11;++d)
		{
				new File(System.getProperty("user.home")+"output/crossvalidation_outputs/fold"+d).mkdir()
		}
		double sumf1 = 0.0;
		for(int i = 0; i <10; ++i)
		{
			UnsupervisedFineThread a = new UnsupervisedFineThread()
			double f1 =a.pslmodel(i);
			sumf1 += f1;
		}
		System.out.println("average f1 "+sumf1/10);
	}

	double pslmodel(int cvSet)
	{
		int folds = 10 // number of folds
		double seedRatio = 1 // ratio of observed labels
		Random rand = new Random(0) // used to seed observed data
		double trainTestRatio = 0.9 // ratio of train to test splits (random)
		double filterRatio = 1.0 // ratio of documents to keep (throw away the rest)
		//targetSize = 3000 // target size of snowball sampler
		double explore = 0.001 // prob of random node in snowball sampler
		int numCategories = 7  // number of label categories
		ConfigManager cm = ConfigManager.getManager()
		ConfigBundle config = cm.getBundle("mooc-aspect-test")
		Logger log = LoggerFactory.getLogger(this.class)

		/* Uses H2 as a DataStore and stores it in a temp. directory by default */
		def defaultPath = System.getProperty("java.io.tmpdir")
		String dbpath = config.getString("dbpath", defaultPath + File.separator + "mooc-aspect-test")
		DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)
		String writefolder = "output/crossvalidation_outputs/"
		PSLModel m = new PSLModel(this, data)
		Map<String, List<DiscretePredictionStatistics>> results = new HashMap<String, List<DiscretePredictionStatistics>>()
		results.put(config, new ArrayList<DiscretePredictionStatistics>())

		//Predicates
		
		
		
		
		
		//Target Predicates
		m.add predicate: "relevance" , types: [ArgumentType.UniqueID]
		m.add predicate: "coarsetopic" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "finetopic_1" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "finetopic_12" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "sentiment" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		
		//fineseeded
		m.add predicate: "fineseeded_1" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_2" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_3" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_4" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_5" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_6" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_7" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_8" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_9" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_10" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_11" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_12" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_13" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_14" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_15" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_16" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_17" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_18" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_19" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_20" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_21" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_22" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_23" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_24" , types: [ArgumentType.UniqueID]
		m.add predicate: "fineseeded_25" , types: [ArgumentType.UniqueID]
		
		//fineseeded2
		
		m.add predicate: "fine2_seeded_1" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine2_seeded_2" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine2_seeded_3" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine2_seeded_4" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine2_seeded_5" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine2_seeded_6" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine2_seeded_7" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine2_seeded_8" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine2_seeded_9" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine2_seeded_10" , types: [ArgumentType.UniqueID]
		
		//sum features
		
		m.add predicate: "lecture_sum" , types: [ArgumentType.UniqueID]
		m.add predicate: "quiz_sum" , types: [ArgumentType.UniqueID]
		
		//sentiment
		m.add predicate: "sentimentseeded_1" , types: [ArgumentType.UniqueID]
		m.add predicate: "sentimentseeded_2" , types: [ArgumentType.UniqueID]
		m.add predicate: "sentimentseeded_3" , types: [ArgumentType.UniqueID]
		m.add predicate: "sentimentseeded_4" , types: [ArgumentType.UniqueID]
		m.add predicate: "sentimentseeded_5" , types: [ArgumentType.UniqueID]
		m.add predicate: "sentimentseeded_6" , types: [ArgumentType.UniqueID]
		
		m.add predicate: "sentimentseeded3_1" , types: [ArgumentType.UniqueID]
		m.add predicate: "sentimentseeded3_2" , types: [ArgumentType.UniqueID]
		m.add predicate: "sentimentseeded3_3" , types: [ArgumentType.UniqueID]
		m.add predicate: "sentimentseeded3_max" , types: [ArgumentType.UniqueID,ArgumentType.UniqueID]
		
		
		
		m.add predicate: "wekasentiment_1" , types: [ArgumentType.UniqueID]
		m.add predicate: "wekasentiment_2" , types: [ArgumentType.UniqueID]
		m.add predicate: "wekasentiment_3" , types: [ArgumentType.UniqueID]

		
		
		m.add predicate: "post" , types: [ArgumentType.UniqueID]
		m.add predicate: "postcategory" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "postfinecategory1" , types: [ArgumentType.UniqueID,ArgumentType.UniqueID]
		m.add predicate: "coarsetopic" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "finetopic_1" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "sentiment_1" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "postsentimentcategory" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "finetopic_2" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		
		m.add predicate: "senti_pos" , types: [ArgumentType.UniqueID]
		m.add predicate: "senti_neg" , types: [ArgumentType.UniqueID]
		m.add predicate: "sentimax" , types: [ArgumentType.UniqueID,ArgumentType.UniqueID]
	
		m.add predicate: "fine1_negative_sum" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine2_negative_sum" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine1_neutral_sum" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine2_neutral_sum" , types: [ArgumentType.UniqueID]
		
		m.add predicate: "fine2_availability_sum" , types: [ArgumentType.UniqueID]
		m.add predicate: "fine1_content_sum" , types: [ArgumentType.UniqueID]

		m.add predicate: "child", types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		
		m.add predicate: "thread" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		m.add predicate: "course" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
		

		/*
		 * Adding rules
		 */

//		data_labels = {
//			"lecture-audio" :	1,
//			"lecture-video" :	2,
//			"lecture-availability" :	3,
//			"lecture-content" :	4,
//			"lecture-lecturer" :	5,
//			"lecture-subtitles" :	6,
//			"quiz-availability"	: 7,
//			"quiz-deadlines"	: 8,
//			"quiz-content"	: 9,
//			"quiz-grading"	: 10,
//			"quiz-submission"	: 11,
//			"certificate"	: 12,
//			"schedule"	: 13,
//			"supplements" : 14,
//			"social"	: 15
//			}

		
		
		
		
		
		
		
		
//		data_labels = {
//			"positive" :	1,
//			"negative" :	2,
//			"neutral" :	3
//			}
//
//		data_labels = {
//			"content" :	1,
//			"correctness" :	2,
//			"availability" :	3,
//			"difficulty" : 4
//			"none":0
//			}
		
		
		m.add rule : (fineseeded_1(P))	>>	finetopic_1(P, data.getUniqueID(2)), weight : 5, squared:true
		m.add rule : (fineseeded_2(P))	>>	finetopic_1(P, data.getUniqueID(1)), weight : 5, squared:true
		m.add rule : (fineseeded_3(P)) >> finetopic_1(P, data.getUniqueID(5)), weight : 5, squared:true
		m.add rule : (fineseeded_4(P)) >> finetopic_1(P, data.getUniqueID(6)), weight : 5, squared:true
		m.add rule : (fineseeded_5(P)) >> finetopic_1(P, data.getUniqueID(3)), weight : 5, squared:true
		m.add rule : (fineseeded_6(P)) >> finetopic_1(P, data.getUniqueID(4)), weight : 5, squared:true
		m.add rule : (fineseeded_7(P)) >> finetopic_1(P, data.getUniqueID(9)), weight : 5, squared:true
		m.add rule : (fineseeded_8(P)) >> finetopic_1(P, data.getUniqueID(11)), weight : 5, squared:true
		m.add rule : (fineseeded_9(P)) >> finetopic_1(P, data.getUniqueID(10)), weight : 5, squared:true
		m.add rule : (fineseeded_10(P)) >> finetopic_1(P, data.getUniqueID(8)), weight : 5, squared:true
		m.add rule : (fineseeded_11(P)) >> finetopic_1(P, data.getUniqueID(12)), weight : 5, squared:true
		m.add rule : (fineseeded_12(P)) >> finetopic_1(P, data.getUniqueID(14)), weight : 5, squared:true
		m.add rule : (fineseeded_13(P)) >> finetopic_1(P, data.getUniqueID(13)), weight : 5, squared:true
		m.add rule : (fineseeded_14(P)) >> finetopic_1(P, data.getUniqueID(15)), weight : 5, squared:true
		m.add rule : (fineseeded_15(P)) >> finetopic_1(P, data.getUniqueID(4)), weight : 5, squared:true
		m.add rule : (fineseeded_16(P)) >> finetopic_1(P, data.getUniqueID(4)), weight : 5, squared:true
		m.add rule : (fineseeded_17(P)) >> finetopic_1(P, data.getUniqueID(4)), weight : 5, squared:true
		m.add rule : (fineseeded_18(P)) >> finetopic_1(P, data.getUniqueID(4)), weight : 5, squared:true
		m.add rule : (fineseeded_15(P)) >> finetopic_1(P, data.getUniqueID(9)), weight : 5, squared:true
		m.add rule : (fineseeded_16(P)) >> finetopic_1(P, data.getUniqueID(9)), weight : 5, squared:true
		m.add rule : (fineseeded_17(P)) >> finetopic_1(P, data.getUniqueID(9)), weight : 5, squared:true
		m.add rule : (fineseeded_18(P)) >> finetopic_1(P, data.getUniqueID(9)), weight : 5, squared:true
//
//
//		//fine2_seeded_1
//
//		m.add rule : (fine2_seeded_1(P)) >> finetopic_1(P, data.getUniqueID(1)), weight : 5, squared:true
//		m.add rule : (fine2_seeded_1(P)) >> finetopic_2(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (fine2_seeded_1(P)) >> finetopic_2(P, data.getUniqueID(3)), weight : 5, squared:true
//		m.add rule : (fine2_seeded_1(P)) >> finetopic_2(P, data.getUniqueID(4)), weight : 5, squared:true
//		m.add rule : (fine2_seeded_1(P)) >> finetopic_2(P, data.getUniqueID(0)), weight : 5, squared:true
//
		
		
		m.add rule : (fineseeded_1(P) & finetopic_1(P, data.getUniqueID(2)))	>>	sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_2(P) & finetopic_1(P, data.getUniqueID(1)))	>>	sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_3(P) & finetopic_1(P, data.getUniqueID(5))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_4(P) & finetopic_1(P, data.getUniqueID(6))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_5(P) & finetopic_1(P, data.getUniqueID(3))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_6(P) & finetopic_1(P, data.getUniqueID(4))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_7(P) & finetopic_1(P, data.getUniqueID(9))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_8(P) & finetopic_1(P, data.getUniqueID(11))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_9(P) & finetopic_1(P, data.getUniqueID(10))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_10(P) & finetopic_1(P, data.getUniqueID(8))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (fineseeded_11(P) & finetopic_1(P, data.getUniqueID(12))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (fineseeded_12(P) & finetopic_1(P, data.getUniqueID(14))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (fineseeded_13(P) & finetopic_1(P, data.getUniqueID(13))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_14(P) & finetopic_1(P, data.getUniqueID(15))) >> sentiment_1(P, data.getUniqueID(1)), weight : 10, squared:true
		m.add rule : (fineseeded_14(P) & finetopic_1(P, data.getUniqueID(15))) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
//		m.add rule : (fineseeded_15(P) & finetopic_1(P, data.getUniqueID(4))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (fineseeded_16(P) & finetopic_1(P, data.getUniqueID(4))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (fineseeded_17(P) & finetopic_1(P, data.getUniqueID(4))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (fineseeded_18(P) & finetopic_1(P, data.getUniqueID(4))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (fineseeded_15(P) & finetopic_1(P, data.getUniqueID(9))) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
		m.add rule : (fineseeded_16(P) & finetopic_1(P, data.getUniqueID(9))) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
		m.add rule : (fineseeded_17(P) & finetopic_1(P, data.getUniqueID(9))) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
		m.add rule : (fineseeded_18(P) & finetopic_1(P, data.getUniqueID(9))) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
		
		
		//Sentiment : 1 - positive, 2 - negative, 3 - neutral
		
		
//		m.add rule : ( sentimentseeded_2(P)& finetopic_1(P, data.getUniqueID(2)))	>>	sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(1)))	>>	sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(5))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(6))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(3))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(4))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(9))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(11))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(10))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(8))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(12))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(14))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(13))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_1(P) & finetopic_1(P, data.getUniqueID(15))) >> sentiment_1(P, data.getUniqueID(1)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_3(P) & finetopic_1(P, data.getUniqueID(15))) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(4))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(4))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(4))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_2(P) & finetopic_1(P, data.getUniqueID(4))) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_3(P) & finetopic_1(P, data.getUniqueID(9))) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_3(P) & finetopic_1(P, data.getUniqueID(9))) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_3(P) & finetopic_1(P, data.getUniqueID(9))) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
//		m.add rule : (sentimentseeded_3(P) & finetopic_1(P, data.getUniqueID(9))) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
		
//		m.add rule : (sentimentseeded_1(P) ) >> sentiment_1(P, data.getUniqueID(1)), weight : 5, squared:true
//		m.add rule : (sentimentseeded_2(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (sentimentseeded_3(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true
		

		
				
//		m.add rule : (wekasentiment_1(P)) >> sentiment_1(P, data.getUniqueID(1)), weight : 5, squared:true
//		m.add rule : (wekasentiment_2(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (wekasentiment_3(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true

		
		
		
		m.add rule : (sentimentseeded3_1(P)) >> sentiment_1(P, data.getUniqueID(1)), weight : 5, squared:true
		m.add rule : (sentimentseeded3_2(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
		m.add rule : (sentimentseeded3_3(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true

		
		
		
//
		
		m.add rule : (fineseeded_1(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_2(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_3(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_4(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_5(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_6(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_7(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_8(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_9(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_10(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
		m.add rule : (fineseeded_11(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true

		
		//m.add rule : (fineseeded_15(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 3, squared:true
//		m.add rule : (fineseeded_15(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
		
//		m.add rule : (fineseeded_16(P) ) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
//		m.add rule : (fineseeded_17(P) ) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
//		m.add rule : (fineseeded_18(P) ) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true
		
//		m.add rule : (fineseeded_13(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
//
//		m.add rule : (fineseeded_14(P)) >> sentiment_1(P, data.getUniqueID(1)), weight : 5, squared:true
		
		
		m.add rule : (fine1_negative_sum(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
		m.add rule : (fine2_negative_sum(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (fine1_neutral_sum(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true
//		m.add rule : (fine2_neutral_sum(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true
//
//		m.add rule : (fine1_content_sum(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (fine2_availability_sum(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
		
//		m.add rule : (fine2_availability_sum(P)) >> finetopic_1(P, data.getUniqueID(3)), weight : 5, squared:true
//		m.add rule : (fine1_content_sum(P)) >> finetopic_1(P, data.getUniqueID(4)), weight : 5, squared:true
//		
//		m.add rule : (sentiment_1(P, data.getUniqueID(1))) >> ~sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (sentiment_1(P, data.getUniqueID(2))) >> ~sentiment_1(P, data.getUniqueID(1)), weight : 5, squared:true
//		
		
//		m.add rule : (fineseeded_15(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true
//		m.add rule : (fineseeded_16(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true
//		m.add rule : (fineseeded_17(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true
//		m.add rule : (fineseeded_18(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true
		
		
		
		
		// Sentiwordnet
		
		m.add rule : (senti_pos(P)) >> sentiment_1(P, data.getUniqueID(1)), weight : 5, squared:true
		m.add rule : (senti_neg(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
		m.add rule : (sentimax(P,data.getUniqueID(1))) >> sentiment_1(P, data.getUniqueID(1)), weight : 5, squared:true
		m.add rule : (sentimax(P,data.getUniqueID(2))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
		m.add rule : (sentimax(P,data.getUniqueID(3))) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true
		
		
		
//		m.add rule : (finetopic_1(P, data.getUniqueID(2))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(1))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(5) ))>> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(6))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(3))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(4))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(9))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(11))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(10))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(8))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(12))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(14))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(13))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(15))) >> sentiment_1(P, data.getUniqueID(1)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(4))) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true
//		m.add rule : (finetopic_1(P, data.getUniqueID(9))) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true
		
		m.add rule : (fine2_seeded_1(P)) >> finetopic_1(P, data.getUniqueID(5)), weight : 5, squared:true
		m.add rule : (fine2_seeded_2(P)) >> finetopic_1(P, data.getUniqueID(5)), weight : 5, squared:true
		m.add rule : (fine2_seeded_3(P)) >> finetopic_1(P, data.getUniqueID(11)), weight : 5, squared:true
		m.add rule : (fine2_seeded_4(P) & lecture_sum(P)) >> finetopic_1(P, data.getUniqueID(4)), weight : 5, squared:true
		m.add rule : (fine2_seeded_4(P) & quiz_sum(P)) >> finetopic_1(P, data.getUniqueID(9)), weight : 5, squared:true
//		m.add rule : (fine2_seeded_5(P)) >> finetopic_1(P, data.getUniqueID(0)), weight : 5, squared:true
//		m.add rule : (fine2_seeded_6(P)) >> finetopic_1(P, data.getUniqueID(0)), weight : 5, squared:true
//		m.add rule : (fine2_seeded_7(P)) >> finetopic_1(P, data.getUniqueID(0)), weight : 5, squared:true
//		m.add rule : (fine2_seeded_8(P)) >> finetopic_1(P, data.getUniqueID(0)), weight : 5, squared:true
//		m.add rule : (fine2_seeded_9(P)) >> finetopic_1(P, data.getUniqueID(0)), weight : 5, squared:true
//		m.add rule : (fine2_seeded_10(P)) >> finetopic_1(P, data.getUniqueID(0)), weight : 5, squared:true
	
		//network rules
		
		//m.add rule : (finetopic_1(P1, F) & thread(P1, T) & course(P1, C) & thread(P2, T) & course(P2, C)) >> finetopic_1(P2, F), weight : 10, squared:true
		
		List<Partition> trainPartition = new ArrayList<Partition>(folds)
		List<Partition> trainLabelsPartition = new ArrayList<Partition>(folds)
		List<Partition> testDataPartition = new ArrayList<Partition>(folds)
		List<Partition> testLabelsPartition = new ArrayList<Partition>(folds)

		/*
		 * Initialize partitions for all cross validation sets
		 */
		for(int initset =0 ;initset<10;++initset)
		{
			trainPartition.add(initset, new Partition(initset))
			trainLabelsPartition.add(initset, new Partition(initset + folds))
			testDataPartition.add(initset, new Partition(initset + 2*folds))
			testLabelsPartition.add(initset, new Partition(initset + 3*folds))
		}

		File file1 = new File(writefolder+"results.txt");
		//		File file2 = new File(filename2);
		File modelfile = new File(writefolder + "model.txt")

		Map<CompatibilityKernel,Weight> weights = new HashMap<CompatibilityKernel, Weight>()
		for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
			weights.put(k, k.getWeight());
		/*
		 * Train data partition, each partition has 9 folders, one kept aside for testing...
		 *
		 * loading the predicates from the data files into the trainPartition
		 */
		String filename
		Integer trainSet
		int countloadchild=0
		for (trainSet = 1 ; trainSet<=10;++trainSet)
		{
		
			String filepath = ""
			filename = 'data'+java.io.File.separator+'4144folds_new'+java.io.File.separator+'fold'+trainSet+java.io.File.separator;
			for (Predicate p : [fineseeded_1, fineseeded_2, fineseeded_3, fineseeded_4, fineseeded_5, fineseeded_6, fineseeded_7, fineseeded_8, fineseeded_9,
				fineseeded_10, fineseeded_11, fineseeded_12, fineseeded_13, fineseeded_14, fineseeded_15,fineseeded_16, fineseeded_17, fineseeded_18, fineseeded_19,
				sentimentseeded_1, sentimentseeded_2, sentimentseeded_3, sentimentseeded_4, sentimentseeded_5,
				sentimentseeded_6, senti_pos, senti_neg,sentimax,fine1_negative_sum,fine2_negative_sum,fine1_neutral_sum,fine2_neutral_sum,fine2_availability_sum,fine1_content_sum,
				sentimentseeded3_1,sentimentseeded3_2,sentimentseeded3_3,wekasentiment_1,wekasentiment_2,wekasentiment_3,
				lecture_sum, quiz_sum])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
				InserterUtils.loadDelimitedDataTruth(data.getInserter(p, trainPartition.get(cvSet)),
						filepath,"\t");
			}
			for (Predicate p : [post, postfinecategory1, postsentimentcategory,sentimentseeded3_max])
			{
				filepath = filename+p.getName().toString().toLowerCase()+".txt"
				println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
				InserterUtils.loadDelimitedData(data.getInserter(p, trainPartition.get(cvSet)),
						filepath,"\t");

			}
			if(countloadchild==0){
				String childfile = 'data'+java.io.File.separator+"child.txt";
				println "\t\t\tREADING child" +" from "+childfile+"...";
				InserterUtils.loadDelimitedData(data.getInserter(child, trainPartition.get(cvSet)),
						childfile,"\t");
				countloadchild = 1
			}

			/*
			 * Load in the ground truth labels for train partition
			 */
			String finetopic_1path = filename+"finetopic_1.txt"
			InserterUtils.loadDelimitedData(data.getInserter(finetopic_1, trainLabelsPartition.get(cvSet)),
					finetopic_1path);
			String sentiment_1path = filename+"sentiment_1.txt"
			InserterUtils.loadDelimitedData(data.getInserter(sentiment_1, trainLabelsPartition.get(cvSet)),
						sentiment_1path);
				
				
				
		}
		filename = 'data'+java.io.File.separator+"unsupervised"+java.io.File.separator
		
		for (Predicate p : [thread, course])
		{
			String filepath = filename+p.getName().toString().toLowerCase()+".txt"
			println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
			InserterUtils.loadDelimitedData(data.getInserter(p, trainPartition.get(cvSet)),
					filepath,"\t");

		}

		Set toClose = [fineseeded_1, fineseeded_2, fineseeded_3, fineseeded_4, fineseeded_5, fineseeded_6, fineseeded_7,
			fineseeded_8, fineseeded_9,fineseeded_10, fineseeded_11, fineseeded_12, fineseeded_13, fineseeded_14,
			fineseeded_15, fineseeded_16, fineseeded_17, fineseeded_18, fineseeded_19, postfinecategory1, post, child, sentimentseeded_1, sentimentseeded_2, sentimentseeded_3,
			sentimentseeded_4, sentimentseeded_5,sentimentseeded_6,postsentimentcategory, senti_pos, senti_neg,sentimax,
			fine2_availability_sum,fine1_content_sum,fine1_negative_sum,fine2_negative_sum,fine1_neutral_sum,fine2_neutral_sum,sentimentseeded3_1,sentimentseeded3_2,sentimentseeded3_3,
			wekasentiment_1,wekasentiment_2,wekasentiment_3,sentimentseeded3_max, lecture_sum, quiz_sum, thread, course] as Set;

		Database trainDB = data.getDatabase(trainPartition.get(cvSet), toClose as Set);
		Database trainLabelsDB = data.getDatabase(trainLabelsPartition.get(cvSet), [finetopic_1,sentiment_1] as Set);

		ResultList allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(postfinecategory1))
		println "groundings for all "+ allGroundings.size();
		for (int j = 0; j < allGroundings.size(); j++) {
			GroundTerm [] grounding = allGroundings.get(j)
			RandomVariableAtom atom1 = trainDB.getAtom(finetopic_1, grounding);
			atom1.setValue(0.0);
			atom1.commitToDB();
			
		}
		allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(postsentimentcategory))
		println "groundings for all "+ allGroundings.size();
		for (int j = 0; j < allGroundings.size(); j++) {
			GroundTerm [] grounding = allGroundings.get(j)
		RandomVariableAtom atom2 = trainDB.getAtom(sentiment_1, grounding);
		atom2.setValue(0.0);
		atom2.commitToDB();
		}
		

		for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
			k.setWeight(weights.get(k))
		//weight learning
		//DualEM weightLearning = new DualEM(m, trainDB, trainLabelsDB, config);
//		MaxLikelihoodMPE weightLearning = new MaxLikelihoodMPE(m, trainDB, trainLabelsDB, config);
//		
//		weightLearning.learn();
//		weightLearning.close();
//		modelfile.append(m)
//		modelfile.append("\n")

		
		MPEInference mpe = new MPEInference(m, trainDB, config)
		FullInferenceResult result = mpe.mpeInference()
		System.out.println("Objective: " + result.getTotalWeightedIncompatibility())

		for (GroundAtom atom : Queries.getAllAtoms(trainDB, finetopic_1)){
			file1.append( atom.toString() + "\t" + atom.getValue()+"\n");
		}

		allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(finetopic_1))
		System.out.println("printing no of fine topic atoms in test db " +allGroundings.size());
		
		allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(sentiment_1))
		System.out.println("printing no of sentiment atoms in test db " +allGroundings.size());
		
		trainLabelsDB.close()
		def groundTruthDB = data.getDatabase(trainLabelsPartition.get(cvSet), [finetopic_1, sentiment_1] as Set)
		DataOutputter.outputPredicate("output" + "/groundTruth" + cvSet + ".finetopicnode" , groundTruthDB, finetopic_1, "\t", false, "nodeid\tlabel")
		DataOutputter.outputPredicate("output" + "/groundTruth" + cvSet + ".sentimentnode" , groundTruthDB, sentiment_1, "\t", false, "nodeid\tlabel")
		
		allGroundings = groundTruthDB.executeQuery(Queries.getQueryForAllAtoms(finetopic_1))
		System.out.println("printing no of finetopic atoms in groundtruth db " +allGroundings.size());
		
		allGroundings = groundTruthDB.executeQuery(Queries.getQueryForAllAtoms(sentiment_1))
		System.out.println("printing no of sentiment atoms in groundtruth db " +allGroundings.size());
		int totalTestExamples = allGroundings.size()
		DataOutputter.outputClassificationPredictions("output" + "/results" + config.getString("name", "") + cvSet + ".txt",
				trainDB, finetopic_1, "\t")
		DataOutputter.outputClassificationPredictions("output" + "/results" + config.getString("name", "") + cvSet + ".txt",
			trainDB, sentiment_1, "\t")


		
		allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(sentiment_1))
		for(int k = 0;k<allGroundings.size();++k){
			GroundTerm [] grounding = allGroundings.get(k)
			System.out.println(grounding);
		}
		HashMap <GroundTerm,Integer> map = new HashMap<GroundTerm, Integer>()
		for(int k = 0;k<allGroundings.size();++k){
			GroundTerm [] grounding = allGroundings.get(k)
			String index = grounding[1].toString()
			if(!map.containsKey(grounding[1])){
				map.put(grounding[1], Integer.parseInt(index))}
		}
		
		def comparator = new MulticlassPredictionComparator(trainDB)
		comparator.setBaseline(groundTruthDB)
		comparator.setResultFilter(new MaxValueFilter(sentiment_1, 1))
		//comparator.setThreshold(Double.MIN_VALUE)
		MulticlassPredictionStatistics stats = comparator.compare(sentiment_1, map,1)
		File outputstats = new File("output/crossvalidation_outputs/stats"+cvSet+".txt")
		println stats.getAccuracy()
		
		outputstats.append("adding sentimentseeded3 max, alongwith sentiment seeded with slack rules")
		println "confusion matrix"+ stats.getConfusionMatrix()
		println "f1 for sentiment_1 " + stats.getF1()
		outputstats.append(  "confusion matrix"+ stats.getConfusionMatrix()+"\n")
		outputstats.append( "f1 for sentiment_0 " + stats.getF1(0)+"\n")
		outputstats.append( "precision for sentiment_0 " + stats.getPrecision(0)+"\n")
		outputstats.append( "recall for sentiment_0 " + stats.getRecall(0)+"\n")

		outputstats.append( "f1 for sentiment_1 " + stats.getF1(1)+"\n")
		outputstats.append( "precision for sentiment_1 " + stats.getPrecision(1)+"\n")
		outputstats.append( "recall for sentiment_1 " + stats.getRecall(1)+"\n")
		
		outputstats.append( "f1 for sentiment_2 " + stats.getF1(2)+"\n")
		outputstats.append( "precision for sentiment_2 " + stats.getPrecision(2)+"\n")
		outputstats.append( "recall for sentiment_2 " + stats.getRecall(2)+"\n")
		
		
		
	
		
		
		
		allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(finetopic_1))
		for(int k = 0;k<allGroundings.size();++k){
			GroundTerm [] grounding = allGroundings.get(k)
			System.out.println(grounding);
		}
		map = new HashMap<GroundTerm, Integer>()
		for(int k = 0;k<allGroundings.size();++k){
			GroundTerm [] grounding = allGroundings.get(k)
			String index = grounding[1].toString()
			if(!map.containsKey(grounding[1])){
				map.put(grounding[1], Integer.parseInt(index))}
		}
		
		comparator = new MulticlassPredictionComparator(trainDB)
		comparator.setBaseline(groundTruthDB)
		comparator.setResultFilter(new MaxValueFilter(finetopic_1, 1))
		//comparator.setThreshold(Double.MIN_VALUE)
		stats = comparator.compare(finetopic_1, map,1)
		File outputstats_fine = new File("output/crossvalidation_outputs/stats_fine1"+cvSet+".txt")
		println stats.getAccuracy()
		
		outputstats_fine.append("joint sentiment finetopic")
		println "confusion matrix"+ stats.getConfusionMatrix()
		println "f1 for fientopic " + stats.getF1()
		outputstats_fine.append(  "f1 for fientopic " + stats.getF1()+"\n")
		outputstats_fine.append(  "confusion matrix"+ stats.getConfusionMatrix()+"\n")
		outputstats_fine.append( "f1 for fientopic 0 " + stats.getF1(0)+"\n")
		outputstats_fine.append( "precision for fientopic 0 " + stats.getPrecision(0)+"\n")
		outputstats_fine.append( "recall for fientopic 0 " + stats.getRecall(0)+"\n")

		outputstats_fine.append( "f1 for fientopic 1 " + stats.getF1(1)+"\n")
		outputstats_fine.append( "precision for fientopic 1 " + stats.getPrecision(1)+"\n")
		outputstats_fine.append( "recall for fientopic 1 " + stats.getRecall(1)+"\n")

		outputstats_fine.append( "f1 for fientopic (2) " + stats.getF1(2)+"\n")
		outputstats_fine.append( "precision for fientopic (2) " + stats.getPrecision(2)+"\n")
		outputstats_fine.append( "recall for fientopic (2) " + stats.getRecall(2)+"\n")

		outputstats_fine.append( "f1 for fientopic (3) " + stats.getF1(3)+"\n")
		outputstats_fine.append( "precision for fientopic (3) " + stats.getPrecision(3)+"\n")
		outputstats_fine.append( "recall for fientopic (3) " + stats.getRecall(3)+"\n")

		outputstats_fine.append( "f1 for finetopic_1 (4) " + stats.getF1(4)+"\n")
		outputstats_fine.append( "precision for finetopic_1 (4) " + stats.getPrecision(4)+"\n")
		outputstats_fine.append( "recall for finetopic_1 (4) " + stats.getRecall(4)+"\n")

		outputstats_fine.append( "f1 for finetopic_1 (5) " + stats.getF1(5)+"\n")
		outputstats_fine.append( "precision for finetopic_1 (5) " + stats.getPrecision(5)+"\n")
		outputstats_fine.append( "recall for finetopic_1 (5) " + stats.getRecall(5)+"\n")

		outputstats_fine.append( "f1 for finetopic_1 (6) " + stats.getF1(6)+"\n")
		outputstats_fine.append( "precision for finetopic_1 (6) " + stats.getPrecision(6)+"\n")
		outputstats_fine.append( "recall for finetopic_1 (6) " + stats.getRecall(6)+"\n")

		outputstats_fine.append( "f1 for finetopic_1 (7) " + stats.getF1(7)+"\n")
		outputstats_fine.append( "precision for finetopic_1 (7) " + stats.getPrecision(7)+"\n")
		outputstats_fine.append( "recall for finetopic_1 (7) " + stats.getRecall(7)+"\n")

		outputstats_fine.append( "f1 for finetopic_1 (8) " + stats.getF1(8)+"\n")
		outputstats_fine.append( "precision for finetopic_1 (8) " + stats.getPrecision(8)+"\n")
		outputstats_fine.append( "recall for finetopic_1 (8) " + stats.getRecall(8)+"\n")

		outputstats_fine.append( "f1 for finetopic_1 (9) " + stats.getF1(9)+"\n")
		outputstats_fine.append( "precision for finetopic_1 (9) " + stats.getPrecision(9)+"\n")
		outputstats_fine.append( "recall for finetopic_1 (9) " + stats.getRecall(9)+"\n")

		outputstats_fine.append( "f1 for finetopic_1 (10) " + stats.getF1(10)+"\n")
		outputstats_fine.append( "precision for finetopic_1 (10) " + stats.getPrecision(10)+"\n")
		outputstats_fine.append( "recall for finetopic_1 (10) " + stats.getRecall(10)+"\n")

		outputstats_fine.append( "f1 for finetopic_1 (11) " + stats.getF1(11)+"\n")
		outputstats_fine.append( "precision for finetopic_1 (11) " + stats.getPrecision(11)+"\n")
		outputstats_fine.append( "recall for finetopic_1 (11) " + stats.getRecall(11)+"\n")

		outputstats_fine.append( "f1 for finetopic_1 (12) " + stats.getF1(12)+"\n")
		outputstats_fine.append( "precision for finetopic_1 (12) " + stats.getPrecision(12)+"\n")
		outputstats_fine.append( "recall for finetopic_1 (12) " + stats.getRecall(12)+"\n")

		outputstats_fine.append( "f1 for finetopic_1 (13) " + stats.getF1(13)+"\n")
		outputstats_fine.append( "precision for finetopic_1 (13) " + stats.getPrecision(13)+"\n")
		outputstats_fine.append( "recall for finetopic_1 (13) " + stats.getRecall(13)+"\n")

		outputstats_fine.append( "f1 for finetopic_1 (14) " + stats.getF1(14)+"\n")
		outputstats_fine.append( "precision for finetopic_1 (14) " + stats.getPrecision(14)+"\n")
		outputstats_fine.append( "recall for finetopic_1 (14) " + stats.getRecall(14)+"\n")



		//			def comparator = new DiscretePredictionComparator(testDB)
		//
		//			//groundTruthDB = data.getDatabase(testLabelPartitions.get(fold), [coarsetopic] as Set)
		//			comparator.setBaseline(groundTruthDB)
		//			comparator.setResultFilter(new MaxValueFilter(finetopic_1, 1))
		//			comparator.setThreshold(Double.MIN_VALUE) // treat best value as true as long as it is nonzero
		//
		//			System.out.println("totalTestExamples " + totalTestExamples)
		//			DiscretePredictionStatistics stats = comparator.compare(finetopic_1, totalTestExamples)
		//			System.out.println("F1 score positive " + stats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE))
		//
		//			def b = DiscretePredictionStatistics.BinaryClass.POSITIVE
		//			System.out.println("Method " + config.getString("name", "") + ", fold " + cvSet +", acc " + stats.getAccuracy() +
		//					", prec " + stats.getPrecision(b) + ", rec " + stats.getRecall(b) +
		//					", F1 " + stats.getF1(b) + ", correct " + stats.getCorrectAtoms().size() +
		//					", tp " + stats.tp + ", fp " + stats.fp + ", tn " + stats.tn + ", fn " + stats.fn)
		//
		//
		//			results.get(config).add(cvSet, stats)
		//

		groundTruthDB.close()
		trainDB.close()
		return(stats.getF1())
		 
	}
}

