//
//  LogisticRegressionLearner.swift
//  TransformationPipeline
//
//  Created by Jacopo Mangiavacchi on 2018.
//  Copyright © 2018 JacopoMangia. All rights reserved.
//

import Foundation
import SwiftPipeline

// LogisticRegression Learner (Classifier)
public struct LogisticRegressionLearner : LearnerProtocol {
    public var info: LearnerInfo
    
    public init(name: String = "LogisticRegressionLearner") {
        //TODO: Parametrize normalization, learningRate, maxSteps !!!

        self.info = LearnerInfo(name: name, type: type(of: self), multiClass: false, multiLabel: false)
    }

    public func train(trainFeatures: MatrixDataFloat, testFeatures: MatrixDataFloat, trainLabels: [[Int]], testLabels: [[Int]]) throws -> (trainResult: TrainResult, modelData: Data) {
        //TODO: Guard if multi labels but info.multiLabels = false throw error

        //Training
        let regression = LogisticRegression()
        regression.normalization = true
        let xMat = trainFeatures.map{$0.map(Double.init)}
        let yVec = trainLabels.map{Double($0[0])}
        regression.train(xMat: xMat, yVec: yVec, learningRate: 0.1, maxSteps: 10000)

        //Validation
        let testXMat = testFeatures.map{$0.map(Double.init)}
        let test_y = testLabels.map{Double($0[0])}
        let pred_y = regression.predict(xMat: testXMat)

        //Get Result
        let distance = Euclidean.distance(pred_y, test_y)
        let cost = regression.cost(trueVec: test_y, predictedVec: pred_y)
        print("Euclidean distance: \(distance) - Cost: \(cost)")
        let result = TrainResult(f1: Float(distance), microAvgAccuracy: Float(cost), macroAvgAccuracy: Float(cost), precision: [Float](), recall: [Float]())

        //Get Weights
        let modelData = regression.weights.withUnsafeBufferPointer { buffer -> Data in
            return Data(buffer: buffer)
        }

        return (result, modelData)
    }

    public func predict(modelData: Data, features: MatrixDataFloat) throws -> PredictResult {
        //TODO: Predict

        //print("=== Predict with: \(String(decoding: modelData, as: UTF8.self))")

        var labels = [[Int]]()
        var confidences = [[Float]]()

        for _ in 0..<features.count {
            labels.append([0, 1])
            let r = Float.random(in: 0..<1)
            confidences.append([r, 1.0 - r])
        }

        return PredictResult(labels: labels, confidences: confidences)
    }
}
