package edu.umd.cs.aspect;

import org.slf4j.Logger
import org.slf4j.LoggerFactory

import com.google.common.collect.Iterables

import util.DataOutputter;
import util.ExperimentConfigGenerator;
import util.FoldUtils;
import util.GroundingWrapper;

import edu.umd.cs.psl.application.inference.LazyMPEInference;
import edu.umd.cs.psl.application.inference.MPEInference
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


/*
 * Config parameters
 */

folds = 5 // number of folds
double seedRatio = 1 // ratio of observed labels
Random rand = new Random(0) // used to seed observed data
trainTestRatio = 0.9 // ratio of train to test splits (random)
filterRatio = 1.0 // ratio of documents to keep (throw away the rest)
//targetSize = 3000 // target size of snowball sampler
explore = 0.001 // prob of random node in snowball sampler
numCategories = 7  // number of label categories


ConfigManager cm = ConfigManager.getManager()
ConfigBundle config = cm.getBundle("mooc-aspect-test")

Logger log = LoggerFactory.getLogger(this.class)

/* Uses H2 as a DataStore and stores it in a temp. directory by default */
def defaultPath = System.getProperty("java.io.tmpdir")
String dbpath = config.getString("dbpath", defaultPath + File.separator + "mooc-aspect-test")
DataStore data = new RDBMSDataStore(new H2DatabaseDriver(Type.Disk, dbpath, true), config)

PSLModel m = new PSLModel(this, data)

//Predicates

m.add predicate: "relevance" , types: [ArgumentType.UniqueID]
m.add predicate: "relevancepred" , types: [ArgumentType.UniqueID]
m.add predicate: "seededLDA_1" , types: [ArgumentType.UniqueID]
m.add predicate: "seededLDA_2" , types: [ArgumentType.UniqueID]
m.add predicate: "seededLDA_3" , types: [ArgumentType.UniqueID]


m.add predicate: "post" , types: [ArgumentType.UniqueID]
m.add predicate: "postcategory" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "coarsepred0" , types: [ArgumentType.UniqueID]
m.add predicate: "coarsepred" , types: [ArgumentType.UniqueID,ArgumentType.UniqueID]
m.add predicate: "coarsepred1" , types: [ArgumentType.UniqueID]
m.add predicate: "coarsepred2" , types: [ArgumentType.UniqueID]
m.add predicate: "coarsepred3" , types: [ArgumentType.UniqueID]
m.add predicate: "coarsepred4" , types: [ArgumentType.UniqueID]
m.add predicate: "coarsepred5" , types: [ArgumentType.UniqueID]
m.add predicate: "coarsepred6" , types: [ArgumentType.UniqueID]
m.add predicate: "coarsetopic" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "finetopic" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID, ArgumentType.UniqueID]
m.add predicate: "postfinecategory" , types: [ArgumentType.UniqueID, ArgumentType.UniqueID,ArgumentType.UniqueID]
m.add predicate: "f_seeded1", types:[ArgumentType.UniqueID]
m.add predicate: "f_seeded2", types:[ArgumentType.UniqueID]
m.add predicate: "f_seeded3", types:[ArgumentType.UniqueID]
m.add predicate: "f_seeded4", types:[ArgumentType.UniqueID]
m.add predicate: "f_seeded5", types:[ArgumentType.UniqueID]
m.add predicate: "f_seeded6", types:[ArgumentType.UniqueID]
m.add predicate: "f_seeded7", types:[ArgumentType.UniqueID]
m.add predicate: "finepred0" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred1" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred2" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred3" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred4" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred5" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred6" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred7" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred8" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred9" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred10" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred11" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred12" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred13" , types: [ArgumentType.UniqueID]
m.add predicate: "finepred14" , types: [ArgumentType.UniqueID]

//Rules


m.add rule : ( post(P) & ~relevancepred(P) ) >> finetopic(P,data.getUniqueID(0), data.getUniqueID(0)),  weight : 10 //, squared: false
//m.add rule : ( seededlda_1(P) ) >> coarsetopic(P, data.getUniqueID(1)),  weight : 5 //, squared: false
//m.add rule : ( seededlda_2(P) ) >> coarsetopic(P, data.getUniqueID(2)),  weight : 5 //, squared: false
//m.add rule : ( seededlda_3(P) ) >> coarsetopic(P, data.getUniqueID(3)),  weight : 5 //, squared: false
//
//
//m.add rule : ( seededlda_1(P) & relevancepred(P)) >> coarsetopic(P, data.getUniqueID(1)),  weight : 6 //, squared: false
//m.add rule : ( seededlda_2(P) & relevancepred(P)) >> coarsetopic(P, data.getUniqueID(2)),  weight : 5 //, squared: false
//m.add rule : ( seededlda_3(P) & relevancepred(P)) >> coarsetopic(P, data.getUniqueID(3)),  weight : 5 //, squared: false
//
//
//m.add rule : ( seededlda_1(P) & coarsepred1(P) ) >> coarsetopic(P, data.getUniqueID(1)),  weight : 5 //, squared: false
//m.add rule : ( seededlda_2(P) & coarsepred2(P) ) >> coarsetopic(P, data.getUniqueID(2)),  weight : 5 //, squared: false
//m.add rule : ( seededlda_3(P) & coarsepred3(P) ) >> coarsetopic(P, data.getUniqueID(3)),  weight : 5 //, squared: false


//m.add rule : ( coarsepred0(P) ) >> coarsetopic(P, data.getUniqueID(0)),  weight : 10 //, squared: false
//m.add rule : ( coarsepred1(P) ) >> coarsetopic(P, data.getUniqueID(1)),  weight : 5 //, squared: false
//m.add rule : ( coarsepred2(P) ) >> coarsetopic(P, data.getUniqueID(2)),  weight : 5 //, squared: false
//m.add rule : ( coarsepred3(P) ) >> coarsetopic(P, data.getUniqueID(3)),  weight : 5 //, squared: false
//m.add rule : ( coarsepred4(P) ) >> coarsetopic(P, data.getUniqueID(4)),  weight : 5 //, squared: false
//m.add rule : ( coarsepred4(P) ) >> coarsetopic(P, data.getUniqueID(5)),  weight : 5 //, squared: false
//m.add rule : ( coarsepred6(P) ) >> coarsetopic(P, data.getUniqueID(6)),  weight : 5 //, squared: false
//
//
//m.add rule : ( seededlda_1(P) & coarsepred(P,data.getUniqueID(1)) ) >> coarsetopic(P, data.getUniqueID(1)),  weight : 5 //, squared: false
//m.add rule : ( seededlda_2(P) & coarsepred(P,data.getUniqueID(2)) ) >> coarsetopic(P, data.getUniqueID(2)),  weight : 5 //, squared: false
//m.add rule : ( seededlda_3(P) & coarsepred(P,data.getUniqueID(3)) ) >> coarsetopic(P, data.getUniqueID(3)),  weight : 5 //, squared: false


/*
 * Binarized coarsepred based on max
 */
//m.add rule : ( coarsepred(P,data.getUniqueID(0)) ) >> coarsetopic(P, data.getUniqueID(0)),  weight : 5 //, squared: false
//
//for (int i = 1; i < 7; i++)  {
//		category = data.getUniqueID(i)
//		m.add rule : (coarsepred(P,category)) >> coarsetopic(P,category),weight: 5.0
//	}


m.add rule : (f_seeded1(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(3)), weight : 5
m.add rule : (f_seeded2(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(4)), weight : 5
m.add rule : (f_seeded3(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(12)), weight : 5
m.add rule : (f_seeded4(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(5)), weight : 5
m.add rule : (f_seeded5(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(6)), weight : 5
m.add rule : (f_seeded7(P) & relevancepred(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(2)), weight : 5


//m.add rule : (coarsepred0(P) & finepred0(P)) >> finetopic(P,data.getUniqueID(0),data.getUniqueID(0)),weight: 5.0
//m.add rule : (coarsepred1(P) & finepred1(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(1)),weight: 5.0
//m.add rule : (coarsepred1(P) & finepred2(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(2)),weight: 5.0
//m.add rule : (coarsepred1(P) & finepred3(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(3)),weight: 5.0
//m.add rule : (coarsepred1(P) & finepred4(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(4)),weight: 5.0
//m.add rule : (coarsepred1(P) & finepred5(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(5)),weight: 5.0
//m.add rule : (coarsepred1(P) & finepred6(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(6)),weight: 5.0
//m.add rule : (coarsepred1(P) & finepred7(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(7)),weight: 5.0
//m.add rule : (coarsepred2(P) & finepred8(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(8)),weight: 5.0
//m.add rule : (coarsepred2(P) & finepred9(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(9)),weight: 5.0
//m.add rule : (coarsepred2(P) & finepred10(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(10)),weight: 5.0
//m.add rule : (coarsepred2(P) & finepred11(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(11)),weight: 5.0
//m.add rule : (coarsepred2(P) & finepred12(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(12)),weight: 5.0
//m.add rule : (coarsepred2(P) & finepred13(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(13)),weight: 5.0
//m.add rule : (coarsepred2(P) & finepred14(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(14)),weight: 5.0


m.add rule : (finepred0(P)) >> finetopic(P,data.getUniqueID(0),data.getUniqueID(0)),weight: 5.0
m.add rule : (finepred1(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(1)),weight: 5.0
m.add rule : (finepred2(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(2)),weight: 5.0
m.add rule : (finepred3(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(3)),weight: 5.0
m.add rule : (finepred4(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(4)),weight: 5.0
m.add rule : (finepred5(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(5)),weight: 5.0
m.add rule : (finepred6(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(6)),weight: 5.0
m.add rule : (finepred7(P)) >> finetopic(P,data.getUniqueID(1),data.getUniqueID(7)),weight: 5.0
m.add rule : (finepred8(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(8)),weight: 5.0
m.add rule : (finepred9(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(9)),weight: 5.0
m.add rule : (finepred10(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(10)),weight: 5.0
m.add rule : (finepred11(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(11)),weight: 5.0
m.add rule : (finepred12(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(12)),weight: 5.0
m.add rule : (finepred13(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(13)),weight: 5.0
m.add rule : (finepred14(P)) >> finetopic(P,data.getUniqueID(2),data.getUniqueID(14)),weight: 5.0





// ensure that HasCat sums to 1
//m.add PredicateConstraint.Functional , on : coarsetopic

Map<CompatibilityKernel,Weight> weights = new HashMap<CompatibilityKernel, Weight>()
for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
	weights.put(k, k.getWeight());
	
def trainDir = 'data'+java.io.File.separator

Partition trainPart = new Partition(0)
Partition targetPart = new Partition(1)

for (Predicate p : [seededLDA_1, seededLDA_2, seededLDA_3])
{
	println "\t\t\tREADING " + p.getName() +" ...";
	insert = data.getInserter(p, trainPart)
	println trainDir+p.getName().toString().toLowerCase()+".txt"
	InserterUtils.loadDelimitedDataTruth(insert, trainDir+java.io.File.separator+"coarse_seeded"+java.io.File.separator+p.getName().toString().toLowerCase()+".txt");
}

for (Predicate p : [relevancepred])
{
	println "\t\t\tREADING " + p.getName() +" ...";
	insert = data.getInserter(p, trainPart)
	println trainDir+p.getName().toString().toLowerCase()+".txt"
	InserterUtils.loadDelimitedDataTruth(insert, trainDir+p.getName().toString().toLowerCase()+".txt");
}
for (Predicate p : [coarsepred])
{
	println "\t\t\tREADING " + p.getName() +" ...";
	insert = data.getInserter(p, trainPart)
	println trainDir+p.getName().toString().toLowerCase()+".txt"
	InserterUtils.loadDelimitedData(insert, trainDir+java.io.File.separator+"coarse_predictions"+java.io.File.separator+p.getName().toString().toLowerCase()+".txt");
}

for (Predicate p : [f_seeded1,f_seeded2,f_seeded3,f_seeded4,f_seeded5,f_seeded6,f_seeded7])
{
	println "\t\t\tREADING " + p.getName() +" ...";
	insert = data.getInserter(p, trainPart)
	println trainDir+"f_seeded"+java.io.File.separator+p.getName().toString().toLowerCase()+".txt"
	InserterUtils.loadDelimitedDataTruth(insert, trainDir+java.io.File.separator+"f_seeded"+java.io.File.separator+p.getName().toString().toLowerCase()+".txt");
}


for (Predicate p : [post, postfinecategory])
{
	println "\t\t\tREADING " + p.getName() +" ...";
	insert = data.getInserter(p, trainPart)
	println trainDir+p.getName().toString().toLowerCase()+".txt"
	InserterUtils.loadDelimitedData(insert, trainDir+p.getName().toString().toLowerCase()+".txt");
}

for (Predicate p : [relevance])
{
	println "\t\t\tREADING " + p.getName() +" ...";
	insert = data.getInserter(p, trainPart)
	println trainDir+p.getName().toString().toLowerCase()+".txt"
	InserterUtils.loadDelimitedDataTruth(insert, trainDir+p.getName().toString().toLowerCase()+".txt");
}

for (Predicate p : [finetopic])
{
	println "\t\t\tREADING " + p.getName() +" ...";
	insert = data.getInserter(p, targetPart)
	println trainDir+p.getName().toString().toLowerCase()+".txt"
	InserterUtils.loadDelimitedData(insert, trainDir+p.getName().toString().toLowerCase()+".txt");
}
for (Predicate p : [finepred0,finepred1,finepred2,finepred3,finepred4,finepred5,finepred6,finepred7,finepred8,finepred9,
	finepred10,finepred11,finepred12,finepred13,finepred14])
{
	
	println "\t\t\tREADING " + p.getName() +" ...";
	insert = data.getInserter(p, trainPart)
	println trainDir+p.getName().toString().toLowerCase()+".txt"
	InserterUtils.loadDelimitedDataTruth(insert, trainDir+java.io.File.separator+"fine_predictions"+java.io.File.separator+p.getName().toString().toLowerCase()+".txt");
	
}
for (Predicate p : [coarsepred0, coarsepred1, coarsepred2, coarsepred3, coarsepred4, coarsepred5, coarsepred6])
{
	
	println "\t\t\tREADING " + p.getName() +" ...";
	insert = data.getInserter(p, trainPart)
	println trainDir+p.getName().toString().toLowerCase()+".txt"
	InserterUtils.loadDelimitedDataTruth(insert, trainDir+java.io.File.separator+"coarse_predictions"+java.io.File.separator+p.getName().toString().toLowerCase()+".txt");
	
}
println m;

trainReadPartitions = new ArrayList<Partition>()
testReadPartitions = new ArrayList<Partition>()
trainWritePartitions = new ArrayList<Partition>()
testWritePartitions = new ArrayList<Partition>()
trainLabelPartitions = new ArrayList<Partition>()
testLabelPartitions = new ArrayList<Partition>()

def keys = new HashSet<Variable>()
ArrayList<Set<Integer>> trainingSeedKeys = new ArrayList<Set<Integer>>()
ArrayList<Set<Integer>> testingSeedKeys = new ArrayList<Set<Integer>>()
ArrayList<Set<Integer>> trainingKeys = new ArrayList<Set<Integer>>()
ArrayList<Set<Integer>> testingKeys = new ArrayList<Set<Integer>>()
def queries = new HashSet<DatabaseQuery>()


/*
 * DEFINE PRIMARY KEY QUERIES FOR FOLD SPLITTING
 */

Variable Post = new Variable("post")
Variable Category = new Variable("category")
keys.add(Post)
keys.add(Category)
queries.add(new DatabaseQuery(finetopic(Post, Category, A).getFormula()))



def partitionDocuments = new HashMap<Partition, Set<GroundTerm>>()

for (int i = 0; i < folds; i++) {
	trainReadPartitions.add(i, new Partition(i + 2))
	testReadPartitions.add(i, new Partition(i + folds + 2))

	trainWritePartitions.add(i, new Partition(i + 2*folds + 2))
	testWritePartitions.add(i, new Partition(i + 3*folds + 2))

	trainLabelPartitions.add(i, new Partition(i + 4*folds + 2))
	testLabelPartitions.add(i, new Partition(i + 5*folds + 2))

	Set<GroundTerm> [] posts = FoldUtils.generateRandomSplit(data, trainTestRatio,
			trainPart, targetPart, trainReadPartitions.get(i),
			testReadPartitions.get(i), trainLabelPartitions.get(i),
			testLabelPartitions.get(i), queries, keys, filterRatio)
	
	partitionDocuments.put(trainReadPartitions.get(i), posts[0])
	partitionDocuments.put(testReadPartitions.get(i), posts[1])
	
	trainingSeedKeys.add(i, new HashSet<Integer>())
	testingSeedKeys.add(i, new HashSet<Integer>())
	trainingKeys.add(i, new HashSet<Integer>())
	testingKeys.add(i, new HashSet<Integer>())

	for (GroundTerm doc : partitionDocuments.get(trainReadPartitions.get(i))) {
		if (rand.nextDouble() < seedRatio)
			trainingSeedKeys.get(i).add(Integer.decode(doc.toString()))
		trainingKeys.get(i).add(Integer.decode(doc.toString()))
	}
	for (GroundTerm doc : partitionDocuments.get(testReadPartitions.get(i))) {
		if (rand.nextDouble() < seedRatio)
			testingSeedKeys.get(i).add(Integer.decode(doc.toString()))
		testingKeys.get(i).add(Integer.decode(doc.toString()))
	}
	//testReadPartitions.add(trainPart)
	//trainReadPartitions.add(trainPart)
	// add all seedKeys into observed partition
	Database db = data.getDatabase(targetPart)
	def trainInserter = data.getInserter(finetopic, trainReadPartitions.get(i))
	def testInserter = data.getInserter(finetopic, testReadPartitions.get(i))
	ResultList res = db.executeQuery(new DatabaseQuery(finetopic(X,Y,Z).getFormula()))
	for (GroundAtom atom : Queries.getAllAtoms(db, finetopic)) {
		Integer atomKey = Integer.decode(atom.getArguments()[0].toString())
		if (trainingSeedKeys.get(i).contains(atomKey)) {
			trainInserter.insertValue(atom.getValue(), atom.getArguments())
		}

		if (testingSeedKeys.get(i).contains(atomKey)) {
			testInserter.insertValue(atom.getValue(), atom.getArguments())
		}
	}
	db.close()
	
	db = data.getDatabase(trainReadPartitions.get(i))
	ResultList list = db.executeQuery(new DatabaseQuery(finetopic(X,Y,Z).getFormula()))
	System.out.println("Instances of coarsetopic: " + list.size()+" "+ trainReadPartitions.get(i))
	db.close()
	db = data.getDatabase(testReadPartitions.get(i))
	list = db.executeQuery(new DatabaseQuery(finetopic(X,Y,Z).getFormula()))
	System.out.println("Instances of coarsetopic: " + list.size() +" "+ testReadPartitions.get(i))
	db.close()
	
	
	db = data.getDatabase(trainLabelPartitions.get(i))
	list = db.executeQuery(new DatabaseQuery(finetopic(X,Y,Z).getFormula()))
	System.out.println("Instances of train label partiions coarsetopic: " + list.size()+" "+ trainLabelPartitions.get(i))
	db.close()
	db = data.getDatabase(testLabelPartitions.get(i))
	list = db.executeQuery(new DatabaseQuery(finetopic(X,Y,Z).getFormula()))
	System.out.println("Instances of test laberl partition coarsetopic: " + list.size() +" "+ testLabelPartitions.get(i))
	db.close()
	
	
	
	
}

Map<String, List<DiscretePredictionStatistics>> results = new HashMap<String, List<DiscretePredictionStatistics>>()
results.put(config, new ArrayList<DiscretePredictionStatistics>())
	
for (int fold = 0; fold < folds; fold++) {
		
			/*** POPULATE DBs ***/
			
			Database db;
			DatabasePopulator dbPop;
			Variable CategoryFine = new Variable("Category")
			Variable CategoryCoarse = new Variable("Category")
			Variable Post1 = new Variable("Post")
			Map<Variable, Set<GroundTerm>> substitutions = new HashMap<Variable, Set<GroundTerm>>()
			
			/* categories */
			Set<GroundTerm> categoryGroundings = new HashSet<GroundTerm>()
			for (int i = 0; i < numCategories; i++)
				categoryGroundings.add(data.getUniqueID(i))
			substitutions.put(CategoryCoarse, categoryGroundings)
			System.out.println(substitutions)
			/* populate HasCat */
			ArrayList<Partition> trainRead = new ArrayList<Partition>();
			ArrayList<Partition> testRead = new ArrayList<Partition>();
			trainRead.add(trainPart)
			testRead.add(trainPart)
			testRead.add(trainLabelPartitions.get(fold))

			toClose = [coarsepred0, coarsepred1, coarsepred2, coarsepred3, coarsepred4, coarsepred5, coarsepred6,
				f_seeded1,f_seeded2,f_seeded3,f_seeded4,f_seeded5,f_seeded6,f_seeded7,
				finepred0,finepred1,finepred2,finepred3,finepred4,finepred5,finepred6,finepred7,finepred8,finepred9,
				finepred10,finepred11,finepred12,finepred13,finepred14,relevance] as Set;
			
			Database trainDB = data.getDatabase(trainWritePartitions.get(fold), toClose, (Partition []) trainRead.toArray())
			Database testDB = data.getDatabase(testWritePartitions.get(fold), toClose, (Partition []) testRead.toArray())
			
			
			int rv = 0, ob = 0
			ResultList allGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(postfinecategory))
			for (int i = 0; i < allGroundings.size(); i++) {
				GroundTerm [] grounding = allGroundings.get(i)
				GroundAtom atom = testDB.getAtom(finetopic, grounding)
				if (atom instanceof RandomVariableAtom) {
					rv++
					testDB.commit((RandomVariableAtom) atom);
				} else
					ob++
			}
			System.out.println("Saw " + rv + " rvs and " + ob + " obs in test DB")

			
			
			
			
			dbPop = new DatabasePopulator(trainDB)
			substitutions.put(Post1, partitionDocuments.get(trainReadPartitions.get(fold)))
			dbPop.populate(new QueryAtom(finetopic, Post1,CategoryCoarse, CategoryFine), substitutions)
			
			dbPop = new DatabasePopulator(testDB)
			substitutions.put(Post1, partitionDocuments.get(testReadPartitions.get(fold)))
			dbPop.populate(new QueryAtom(finetopic, Post1, CategoryCoarse, CategoryFine), substitutions)
			
			Partition dummy = new Partition(402) 
			Partition dummy3 = new Partition(406)
			toClose = [finetopic] as Set
			Database labelsDB = data.getDatabase(dummy, toClose, trainLabelPartitions.get(fold))
			allGroundings = labelsDB.executeQuery(Queries.getQueryForAllAtoms(finetopic))
			System.out.println("printing no of atoms in labelsDB db " +allGroundings.size());

			def groundTruthDB = data.getDatabase(testLabelPartitions.get(fold), [finetopic] as Set)
			DataOutputter.outputPredicate("output" + "/groundTruth" + fold + ".node" , groundTruthDB, finetopic, "\t", false, "nodeid\tlabel")
			
			
			/*
			 * POPULATE TRAINING DATABASE
			 */
			rv = 0; ob = 0;
			allGroundings = trainDB.executeQuery(Queries.getQueryForAllAtoms(postfinecategory))
			for (int i = 0; i < allGroundings.size(); i++) {
				GroundTerm [] grounding = allGroundings.get(i)
				GroundAtom atom = trainDB.getAtom(finetopic, grounding)
				if (atom instanceof RandomVariableAtom) {
					rv++
					trainDB.commit((RandomVariableAtom) atom);
				} else
					ob++
			}
			System.out.println("Saw " + rv + " rvs and " + ob + " obs")
			/*
			 * POPULATE TEST DATABASE
			 */
			allGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(postfinecategory))
			for (int i = 0; i < allGroundings.size(); i++) {
				GroundTerm [] grounding = allGroundings.get(i)
				GroundAtom atom = testDB.getAtom(finetopic, grounding)
				if (atom instanceof RandomVariableAtom) {
					testDB.commit((RandomVariableAtom) atom);
				}
			}
			
			/*** EXPERIMENT ***/
			for (CompatibilityKernel k : Iterables.filter(m.getKernels(), CompatibilityKernel.class))
			k.setWeight(weights.get(k))
			
			MaxLikelihoodMPE mle = new MaxLikelihoodMPE(m, trainDB, labelsDB, config)
			mle.learn()
			mle.close()
			
			System.out.println("Learned model " + config.getString("name", "") + "\n" + m.toString())
			
			
			/* Inference on test set */
			Set<GroundAtom> allAtoms = Queries.getAllAtoms(testDB, finetopic)
			for (RandomVariableAtom atom : Iterables.filter(allAtoms, RandomVariableAtom))
				atom.setValue(0.0)
			/* For discrete MRFs, "MPE" inference will actually perform marginal inference */
			System.out.println(testDB);
			MPEInference mpe = new MPEInference(m, testDB, config)
			FullInferenceResult result = mpe.mpeInference()
			System.out.println("Objective: " + result.getTotalWeightedIncompatibility())
			
//			println "test db"
//			for (GroundAtom atom : Queries.getAllAtoms(testDB, coarsetopic)){
//				println "inference value at test time"+atom.toString() + "\t" + atom.getValue();
//			}
			
			/* Evaluation */
			allGroundings = testDB.executeQuery(Queries.getQueryForAllAtoms(finetopic))
			System.out.println("printing no of atoms in test db " +allGroundings.size());
			allGroundings = groundTruthDB.executeQuery(Queries.getQueryForAllAtoms(finetopic))
			System.out.println("printing no of atoms in groundtruth db " +allGroundings.size());
			

			dummy2 = new Partition(500);
//			Database resultsDB = data.getDatabase(dummy2, testWritePartitions.get(fold))
//			allGroundings = resultsDB.executeQuery(Queries.getQueryForAllAtoms(coarsetopic))
//			System.out.println("printing no of atoms in resultsDB db " +allGroundings.size());
//			allAtoms = Queries.getAllAtoms(testDB, coarsetopic)
//			Set<RandomVariableAtom> reqdAtoms = Iterables.filter(allAtoms, RandomVariableAtom);
//			
//			for (RandomVariableAtom atom : Iterables.filter(allAtoms, RandomVariableAtom))
//				reqdAtoms.add(atom)
				
			testDB.close();
						Database resultsDB = data.getDatabase(dummy2, testWritePartitions.get(fold))
						allGroundings = resultsDB.executeQuery(Queries.getQueryForAllAtoms(finetopic))
						System.out.println("printing no of atoms in resultsDB db " +allGroundings.size());
			
			
			def comparator = new DiscretePredictionComparator(resultsDB)
			
			//groundTruthDB = data.getDatabase(testLabelPartitions.get(fold), [coarsetopic] as Set)
			comparator.setBaseline(groundTruthDB)
			comparator.setResultFilter(new MaxValueFilter(finetopic, 1))
			comparator.setThreshold(Double.MIN_VALUE) // treat best value as true as long as it is nonzero
	
			int totalTestExamples = testingKeys.get(fold).size() * numCategories;
			System.out.println("totalTestExamples " + totalTestExamples)
			DiscretePredictionStatistics stats = comparator.compare(finetopic, totalTestExamples)
			System.out.println("F1 score " + stats.getF1(DiscretePredictionStatistics.BinaryClass.POSITIVE))
	
			results.get(config).add(fold, stats)
				
			DataOutputter.outputClassificationPredictions("output" + "/results" + config.getString("name", "") + fold + ".txt", resultsDB, finetopic, "\t")
			groundTruthDB.close()
//			testDB.close()
			trainDB.close()
			labelsDB.close()
			resultsDB.close()
			
}

def methodStats = results.get(config)
for (int fold = 0; fold < folds; fold++) {
	def stats = methodStats.get(fold)
	def b = DiscretePredictionStatistics.BinaryClass.POSITIVE
	System.out.println("Method " + config.getString("name", "") + ", fold " + fold +", acc " + stats.getAccuracy() +
			", prec " + stats.getPrecision(b) + ", rec " + stats.getRecall(b) +
			", F1 " + stats.getF1(b) + ", correct " + stats.getCorrectAtoms().size() +
			", tp " + stats.tp + ", fp " + stats.fp + ", tn " + stats.tn + ", fn " + stats.fn)
}

