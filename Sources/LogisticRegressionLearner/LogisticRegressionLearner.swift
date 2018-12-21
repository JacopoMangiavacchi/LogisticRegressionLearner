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

        //Converting data
        let xMat = trainFeatures.map{$0.map(Double.init)}
        let yVec = trainLabels.map{Double($0[0])}
        let testXMat = testFeatures.map{$0.map(Double.init)}
        let test_y = testLabels.map{Double($0[0])}

        //Training
        let regression = LogisticRegression()
        regression.normalization = false
        regression.train(xMat: xMat, yVec: yVec, learningRate: 0.1, maxSteps: 10000)

        //Validation
        let pred_y = regression.predict(xMat: testXMat)

        //Get Result
        let cost = regression.cost(trueVec: yVec, predictedVec: regression.predict(xMat: xMat))
        let distance = Euclidean.distance(pred_y, test_y)
        let result = TrainResult(cost: Float(cost), euclideanDistance: Float(distance))

        //TODO: weights Quantization / Compression

        //Get Weights
        let modelData = regression.weights.withUnsafeBufferPointer { buffer -> Data in
            return Data(buffer: buffer)
        }

        return (result, modelData)
    }

    public func predict(modelData: Data, features: MatrixDataFloat) throws -> PredictResult {
        let regression = LogisticRegression()

        //Converting data
        let xMat = features.map{$0.map(Double.init)}

        //TODO: weights De-Quantization / Decompression

        //Set Weights
        regression.normalization = false
        regression.weights = modelData.withUnsafeBytes {
            [Double](UnsafeBufferPointer<Double>(start: $0, count: modelData.count / MemoryLayout<Double>.size))
        }

        //Predict
        let pred_y = regression.predict(xMat: xMat)

        var labels = [[Int]]()
        var confidences = [[Float]]()

        for i in 0..<features.count {
            labels.append([0, 1])
            let r = Float(pred_y[i])
            confidences.append([1.0 - r, r])
        }

        return PredictResult(labels: labels, confidences: confidences)
    }
}
