import XCTest
import SwiftPipeline
@testable import LogisticRegressionLearner

final class LogisticRegressionLearnerTests: XCTestCase {
    func testFakePipeline() {
        let data = ["this  is a    very  long text! . I would, like  to ,  .  translate it", 
                    "today I  earned $1.500 that it's way more than the $ 650 of the other day",
                    "This is a short-text",
                    "another short-text",
                    "and yet another short-text",
                    "I really don't know what to say here",
                    "that is a a document that really duplicate that",
                    "Loving the opening debates rap game Kathy Crowley October 20 - Bruno Mars http://t.co/jyQ3lqbU"]

        let labels = [["label1"], 
                    ["label0"],
                    ["label2", "label0"],
                    ["label2"],
                    ["label2"],
                    ["label0"],
                    ["label1"],
                    ["label0"]]

        let test = ["Sample Text", "second Text"]

        let pipeline = Pipeline(transformers: [//FastText(fastTextModelPath: "/Jacopo/fastText/model.bin"), 
                                            MultiRegex(regexValues: ["\\$\\ ?[+-]?[0-9]{1,3}(?:,?[0-9])*(?:\\.[0-9]{1,2})?"]), 
                                            Tokenizer(separators: " .,!?-", stopWords: ["text", "like"]), 
                                            BOW(name: "Words"), 
                                            BOW(name: "WordGrams3", keyType: .WordGram, gramSize: 3, valueType: .TFIDF(minCount: 1)),
                                            BOW(name: "CharGrams5", keyType: .CharGram, gramSize: 5, valueType: .TFIDF(minCount: 2)),
                                            BOW(name: "HashWords", keyType: .WordGram, gramSize: 1, valueType: .HashingTrick(algorithm: .DJB2, vectorSize: 300)),
                                            MultiDictionary(words: ["long", "big"]), 
                                            BinaryDictionary(words: ["long", "big"])], 
                                    splitRate: 0.3,
                                    minNumberOfRowInSplit: 3)

        var classifier = Classifier(pipeline: pipeline,
                                    learners: [LogisticRegressionLearner()], 
                                    splitTest: 0.3)

        //TRAINING
        try! classifier.train(input: data, labels: labels)
        print(classifier.trainResults)

        //SCORING
        let prediction = try! classifier.predict(input: test) 
        print(prediction)
    }

    static var allTests = [
        ("testFakePipeline", testFakePipeline),
    ]
}
