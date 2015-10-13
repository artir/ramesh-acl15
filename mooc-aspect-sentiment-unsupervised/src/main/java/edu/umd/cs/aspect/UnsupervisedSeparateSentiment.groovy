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


ConfigManager cm = ConfigManager.getManager()
ConfigBundle config = cm.getBundle("mooc-aspect-test")
Logger log = LoggerFactory.getLogger(this.class)

/* Uses H2 as a DataStore and stores it in a temp. directory by default */
def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "mooc-aspect-test")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)
String writefolder = "output/unsupervised/"
PSLModel m = new PSLModel(this, data)

//Predicates

//Target Predicates
m.add predicate: "coarsetopic" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "finetopic_1" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "sentiment" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

//coarseseeded
m.add predicate: "seedldacoarse" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]

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

m.add rule : (seedldacoarse(P, data.getUniqueID(1)))	>>	sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
m.add rule : (seedldacoarse(P, data.getUniqueID(2)))	>>	sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
m.add rule : (seedldacoarse(P, data.getUniqueID(3)))	>>	sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
m.add rule : (seedldacoarse(P, data.getUniqueID(4)))	>>	sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
m.add rule : (seedldacoarse(P, data.getUniqueID(5)))	>>	sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true
m.add rule : (seedldacoarse(P, data.getUniqueID(6)))	>>	sentiment_1(P, data.getUniqueID(1)), weight : 5, squared:true
m.add rule : (seedldacoarse(P, data.getUniqueID(6)))	>>	sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true

//
m.add rule : (fine2_seeded_1(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
m.add rule : (fine2_seeded_2(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
m.add rule : (fine2_seeded_3(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
m.add rule : (fine2_seeded_4(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
m.add rule : (fine2_seeded_5(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true

//Sentiment : 1 - positive, 2 - negative, 3 - neutral

//sentimentseeded -> sentiment
m.add rule : (sentimentseeded3_1(P)) >> sentiment_1(P, data.getUniqueID(1)), weight : 5, squared:true
m.add rule : (sentimentseeded3_2(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
m.add rule : (sentimentseeded3_3(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true

//fineseeded -> sentiment
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


//m.add rule : (fineseeded_15(P)) >> sentiment_1(P, data.getUniqueID(1)), weight : 3, squared:true
//m.add rule : (fineseeded_15(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 10, squared:true



		m.add rule : (fineseeded_13(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 10, squared:true

		m.add rule : (fineseeded_14(P)) >> sentiment_1(P, data.getUniqueID(1)), weight : 5, squared:true


m.add rule : (fine1_negative_sum(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
m.add rule : (fine2_negative_sum(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
m.add rule : (fine1_neutral_sum(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true
m.add rule : (fine2_neutral_sum(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true
//
m.add rule : (fine1_content_sum(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
m.add rule : (fine2_availability_sum(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
//
//

m.add rule : (fineseeded_15(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 2, squared:true
m.add rule : (fineseeded_16(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 2, squared:true
m.add rule : (fineseeded_17(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 2, squared:true
m.add rule : (fineseeded_18(P)) >> sentiment_1(P, data.getUniqueID(3)), weight : 2, squared:true


// Sentiwordnet

m.add rule : (senti_pos(P)) >> sentiment_1(P, data.getUniqueID(1)), weight : 5, squared:true
m.add rule : (senti_neg(P)) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
m.add rule : (sentimax(P,data.getUniqueID(1))) >> sentiment_1(P, data.getUniqueID(1)), weight : 5, squared:true
m.add rule : (sentimax(P,data.getUniqueID(2))) >> sentiment_1(P, data.getUniqueID(2)), weight : 5, squared:true
m.add rule : (sentimax(P,data.getUniqueID(3))) >> sentiment_1(P, data.getUniqueID(3)), weight : 5, squared:true

int folds = 10
int cvSet = 1
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
String filename, filename1
Integer trainSet
int countloadchild=0
filename1 = 'data'+java.io.File.separator+'unsupervised'+java.io.File.separator
String filepath = ""
for (Predicate p : [seedldacoarse])
{
	filepath = filename1+p.getName().toString().toLowerCase()+".txt"
	println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
	InserterUtils.loadDelimitedDataTruth(data.getInserter(p, trainPartition.get(cvSet)),
			filepath,"\t");

}

for (Predicate p : [postcategory])
{
	filepath = filename1+p.getName().toString().toLowerCase()+".txt"
	println "\t\t\tREADING " + p.getName() +" from "+filepath+"...";
	InserterUtils.loadDelimitedData(data.getInserter(p, trainPartition.get(cvSet)),
			filepath,"\t");

}
		
for (trainSet = 1 ; trainSet<=10;++trainSet)
{
	
	filename = 'data'+java.io.File.separator+'4144folds_new'+java.io.File.separator+'fold'+trainSet+java.io.File.separator;
	for (Predicate p : [fineseeded_1, fineseeded_2, fineseeded_3, fineseeded_4, fineseeded_5, fineseeded_6, fineseeded_7, fineseeded_8, fineseeded_9,
		fineseeded_10, fineseeded_11, fineseeded_12, fineseeded_13, fineseeded_14, fineseeded_15,fineseeded_16, fineseeded_17, fineseeded_18, fineseeded_19,
		sentimentseeded_1, sentimentseeded_2, sentimentseeded_3, sentimentseeded_4, sentimentseeded_5,
		sentimentseeded_6, senti_pos, senti_neg,sentimax,fine1_negative_sum,fine2_negative_sum,fine1_neutral_sum,fine2_neutral_sum,fine2_availability_sum,fine1_content_sum,
		sentimentseeded3_1,sentimentseeded3_2,sentimentseeded3_3,wekasentiment_1,wekasentiment_2,wekasentiment_3])
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

	
	String sentiment_1path = filename+"sentiment_1.txt"
	InserterUtils.loadDelimitedData(data.getInserter(sentiment_1, trainLabelsPartition.get(cvSet)),
				sentiment_1path);		
}

		Set toClose = [fineseeded_1, fineseeded_2, fineseeded_3, fineseeded_4, fineseeded_5, fineseeded_6, fineseeded_7,
			fineseeded_8, fineseeded_9,fineseeded_10, fineseeded_11, fineseeded_12, fineseeded_13, fineseeded_14,
			fineseeded_15, fineseeded_16, fineseeded_17, fineseeded_18, fineseeded_19, postcategory, postfinecategory1, post, child, sentimentseeded_1, sentimentseeded_2, sentimentseeded_3,
			sentimentseeded_4, sentimentseeded_5,sentimentseeded_6,postsentimentcategory, senti_pos, senti_neg,sentimax,
			fine2_availability_sum,fine1_content_sum,fine1_negative_sum,fine2_negative_sum,fine1_neutral_sum,fine2_neutral_sum,sentimentseeded3_1,sentimentseeded3_2,sentimentseeded3_3,
			wekasentiment_1,wekasentiment_2,wekasentiment_3,sentimentseeded3_max] as Set;

		Database trainDB = data.getDatabase(trainPartition.get(cvSet), toClose as Set);
		Database trainLabelsDB = data.getDatabase(trainLabelsPartition.get(cvSet), [sentiment_1] as Set);

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
		
		MPEInference mpe = new MPEInference(m, trainDB, config)
		FullInferenceResult result = mpe.mpeInference()
		System.out.println("Objective: " + result.getTotalWeightedIncompatibility())
		
		allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(sentiment_1))
		System.out.println("printing no of sentiment atoms in test db " +allGroundings.size());
		
		trainLabelsDB.close()
		def groundTruthDB = data.getDatabase(trainLabelsPartition.get(cvSet), [ sentiment_1] as Set)

		allGroundings = groundTruthDB.executeQuery(Queries.getQueryForAllAtoms(sentiment_1))
		//System.out.println("printing no of sentiment atoms in groundtruth db " +allGroundings.size());
		int totalTestExamples = allGroundings.size()
		DataOutputter.outputClassificationPredictions("output" + "/unsupervised" + config.getString("name", "") + cvSet + ".txt",
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
		File outputstats = new File("output/unsupervised/"+cvSet+".txt")
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

		groundTruthDB.close()
		trainDB.close()

