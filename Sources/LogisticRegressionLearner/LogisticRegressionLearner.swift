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
        self.info = LearnerInfo(name: name, type: type(of: self), multiClass: false, multiLabel: false)
    }

    public func train(trainFeatures: MatrixDataFloat, testFeatures: MatrixDataFloat, trainLabels: [[Int]], testLabels: [[Int]]) throws -> (trainResult: TrainResult, modelData: Data) {
        //TODO: Guard if multi labels but info.multiLabels = false throw error

        //TODO: Split features / dataset
        //TODO: Training
        //TODO: Validation
        //TODO: Get Weights

        let modelData = Data(String("Model Data r=(\(Int.random(in: 0..<99))) !!!").utf8)

        return (TrainResult(), modelData)
    }

    public func predict(modelData: Data, features: MatrixDataFloat) throws -> PredictResult {
        //TODO: Predict

        print("=== Predict with: \(String(decoding: modelData, as: UTF8.self))")

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
