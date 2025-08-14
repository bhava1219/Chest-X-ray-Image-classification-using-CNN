"use client"

import type React from "react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Badge } from "@/components/ui/badge"
import { Upload, FileImage, Brain, AlertCircle, CheckCircle2, Clock, Zap, Activity, Shield } from "lucide-react"
import { Progress } from "@/components/ui/progress"

type ModelType = "rf" | "dt" | "svm" | "cnn"
type PredictionResult = {
  model: string
  inferred_shape: {
    H: number
    W: number
    C: number
    n_features: number
  }
  prediction: string
  inference_ms: number
  probs: [number, number] // [normal_prob, pneumonia_prob]
}

export default function XRayAnalyzer() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [selectedModel, setSelectedModel] = useState<ModelType>("cnn")
  const [isAnalyzing, setIsAnalyzing] = useState(false)
  const [result, setResult] = useState<PredictionResult | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)

  const modelNames = {
    rf: "Random Forest",
    dt: "Decision Tree",
    svm: "Support Vector Machine",
    cnn: "Convolutional Neural Network",
  }

  const handleFileSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (file) {
      if (!file.type.startsWith("image/")) {
        setError("Please select a valid image file")
        return
      }
      setSelectedFile(file)
      setError(null)
      setResult(null)

      // Create preview URL
      const url = URL.createObjectURL(file)
      setPreviewUrl(url)
    }
  }

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Please select an X-ray image first")
      return
    }

    setIsAnalyzing(true)
    setError(null)
    setResult(null)

    try {
      const formData = new FormData()
      formData.append("file", selectedFile)

      const response = await fetch(`https://cnn-model-prediction.onrender.com/predict_image?model=${selectedModel}`, {
        method: "POST",
        body: formData,
      })

      if (!response.ok) {
        throw new Error(`Analysis failed: ${response.statusText}`)
      }

      const data = await response.json()
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : "Analysis failed. Please try again.")
    } finally {
      setIsAnalyzing(false)
    }
  }

  const resetAnalysis = () => {
    setSelectedFile(null)
    setResult(null)
    setError(null)
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
      setPreviewUrl(null)
    }
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-blue-50 to-cyan-50 dark:from-slate-900 dark:via-slate-800 dark:to-slate-900">
      <div className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-r from-blue-600/10 to-cyan-500/10 dark:from-blue-400/5 dark:to-cyan-400/5" />
        <div className="relative max-w-7xl mx-auto px-4 py-16 sm:px-6 lg:px-8">
          <div className="text-center space-y-6">
            <div className="flex items-center justify-center gap-3 mb-6">
              <div className="relative">
                <Brain className="h-12 w-12 text-primary pulse-glow" />
                <div className="absolute inset-0 bg-primary/20 rounded-full blur-xl" />
              </div>
              <h1 className="text-5xl font-serif font-black text-slate-900 dark:text-white tracking-tight">
                X-Ray Pneumonia Analyzer
              </h1>
            </div>
            <p className="text-xl text-slate-600 dark:text-slate-300 max-w-3xl mx-auto leading-relaxed">
              Empower Your Diagnosis: Predict Pneumonia with Confidence
            </p>
            <p className="text-lg text-slate-500 dark:text-slate-400 max-w-2xl mx-auto">
              Utilize advanced X-ray analysis to enhance patient care with state-of-the-art machine learning models.
            </p>

            <div className="flex items-center justify-center gap-8 mt-8 text-sm text-slate-600 dark:text-slate-400">
              <div className="flex items-center gap-2">
                <Zap className="h-4 w-4 text-primary" />
                <span>Fast Analysis</span>
              </div>
              <div className="flex items-center gap-2">
                <Activity className="h-4 w-4 text-primary" />
                <span>Multiple Models</span>
              </div>
              <div className="flex items-center gap-2">
                <Shield className="h-4 w-4 text-primary" />
                <span>Secure & Private</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div className="max-w-7xl mx-auto px-4 pb-16 sm:px-6 lg:px-8">
        <div className="grid lg:grid-cols-2 gap-8">
          <Card className="border-0 shadow-xl bg-white/80 dark:bg-slate-800/80 backdrop-blur-sm">
            <CardHeader className="pb-6">
              <CardTitle className="flex items-center gap-3 text-2xl font-serif font-bold text-slate-900 dark:text-white">
                <div className="p-2 bg-primary/10 rounded-lg">
                  <Upload className="h-6 w-6 text-primary" />
                </div>
                Upload X-Ray Image
              </CardTitle>
              <CardDescription className="text-base text-slate-600 dark:text-slate-400">
                Select a high-quality X-ray image for AI-powered pneumonia analysis
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="relative group">
                <div className="border-2 border-dashed border-slate-300 dark:border-slate-600 rounded-xl p-8 text-center hover:border-primary hover:bg-primary/5 transition-all duration-300 group-hover:shadow-lg">
                  <input type="file" accept="image/*" onChange={handleFileSelect} className="hidden" id="file-upload" />
                  <label htmlFor="file-upload" className="cursor-pointer block">
                    <div className="relative">
                      <FileImage className="h-16 w-16 text-slate-400 group-hover:text-primary mx-auto mb-4 transition-colors" />
                      <div className="absolute inset-0 bg-primary/20 rounded-full blur-xl opacity-0 group-hover:opacity-100 transition-opacity" />
                    </div>
                    <p className="text-lg font-medium text-slate-700 dark:text-slate-300 mb-2">
                      Click to select an X-ray image
                    </p>
                    <p className="text-sm text-slate-500 dark:text-slate-400">
                      Supports JPG, PNG, and other image formats • Max 10MB
                    </p>
                  </label>
                </div>
              </div>

              {selectedFile && (
                <div className="flex items-center justify-between p-4 bg-gradient-to-r from-primary/10 to-accent/10 rounded-xl border border-primary/20">
                  <div className="flex items-center gap-3">
                    <div className="p-2 bg-primary/20 rounded-lg">
                      <FileImage className="h-5 w-5 text-primary" />
                    </div>
                    <div>
                      <p className="font-medium text-slate-900 dark:text-white">{selectedFile.name}</p>
                      <p className="text-sm text-slate-600 dark:text-slate-400">
                        {(selectedFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                  </div>
                  <Button variant="ghost" size="sm" onClick={resetAnalysis} className="hover:bg-white/50">
                    Remove
                  </Button>
                </div>
              )}

              <div className="space-y-3">
                <label className="text-lg font-medium text-slate-900 dark:text-white">Analysis Model</label>
                <Select value={selectedModel} onValueChange={(value: ModelType) => setSelectedModel(value)}>
                  <SelectTrigger className="h-12 text-base border-2 hover:border-primary/50 transition-colors">
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent className="border-2 shadow-2xl">
                    <SelectItem value="cnn" className="py-4 h-auto cursor-pointer hover:bg-primary/5">
                      <div className="flex flex-col gap-1">
                        <span className="font-semibold text-slate-900 dark:text-white">
                          CNN - Convolutional Neural Network
                        </span>
                      </div>
                    </SelectItem>
                    <SelectItem value="rf" className="py-4 h-auto cursor-pointer hover:bg-primary/5">
                      <div className="flex flex-col gap-1">
                        <span className="font-semibold text-slate-900 dark:text-white">Random Forest</span>
                      </div>
                    </SelectItem>
                    <SelectItem value="dt" className="py-4 h-auto cursor-pointer hover:bg-primary/5">
                      <div className="flex flex-col gap-1">
                        <span className="font-semibold text-slate-900 dark:text-white">Decision Tree</span>
                      </div>
                    </SelectItem>
                    <SelectItem value="svm" className="py-4 h-auto cursor-pointer hover:bg-primary/5">
                      <div className="flex flex-col gap-1">
                        <span className="font-semibold text-slate-900 dark:text-white">Support Vector Machine</span>
                      </div>
                    </SelectItem>
                  </SelectContent>
                </Select>
              </div>

              <Button
                onClick={handleAnalyze}
                disabled={!selectedFile || isAnalyzing}
                className="w-full h-14 text-lg font-semibold bg-gradient-to-r from-primary via-blue-600 to-accent hover:from-primary/90 hover:via-blue-600/90 hover:to-accent/90 shadow-lg hover:shadow-2xl transition-all duration-500 transform hover:scale-[1.02] active:scale-[0.98] relative overflow-hidden group"
                size="lg"
              >
                <div className="absolute inset-0 bg-gradient-to-r from-white/0 via-white/20 to-white/0 translate-x-[-100%] group-hover:translate-x-[100%] transition-transform duration-1000" />
                {isAnalyzing ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-3" />
                    <span className="shimmer relative z-10">Analyzing X-Ray...</span>
                  </>
                ) : (
                  <>
                    <Brain className="h-5 w-5 mr-3 relative z-10" />
                    <span className="relative z-10">Start Analysis</span>
                  </>
                )}
              </Button>
            </CardContent>
          </Card>

          <Card className="border-0 shadow-2xl bg-white/90 dark:bg-slate-800/90 backdrop-blur-md relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-primary/10 via-accent/10 to-primary/10 opacity-0 hover:opacity-100 transition-opacity duration-500" />
            <CardHeader className="pb-6 relative z-10">
              <CardTitle className="flex items-center gap-3 text-2xl font-serif font-bold text-slate-900 dark:text-white">
                <div className="p-2 bg-accent/10 rounded-lg relative">
                  <Activity className="h-6 w-6 text-accent" />
                  <div className="absolute inset-0 bg-accent/20 rounded-lg blur-md animate-pulse" />
                </div>
                Analysis Results
              </CardTitle>
              <CardDescription className="text-base text-slate-600 dark:text-slate-400">
                AI prediction results and diagnostic insights
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-6 relative z-10">
              {previewUrl && (
                <div className="space-y-3">
                  <label className="text-lg font-medium text-slate-900 dark:text-white">X-Ray Preview</label>
                  <div className="relative overflow-hidden rounded-2xl border-2 border-slate-200 dark:border-slate-700 shadow-xl group">
                    <img
                      src={previewUrl || "/placeholder.svg"}
                      alt="X-ray preview"
                      className="w-full h-64 object-cover group-hover:scale-110 transition-transform duration-700"
                    />
                    <div className="absolute inset-0 bg-gradient-to-t from-black/40 via-transparent to-transparent" />
                    <div className="absolute top-4 right-4 bg-white/90 dark:bg-slate-800/90 backdrop-blur-sm rounded-full p-2">
                      <FileImage className="h-4 w-4 text-primary" />
                    </div>
                  </div>
                </div>
              )}

              {/* Error Display */}
              {error && (
                <Alert variant="destructive" className="border-red-200 bg-red-50 dark:bg-red-900/20">
                  <AlertCircle className="h-5 w-5" />
                  <AlertDescription className="text-base">{error}</AlertDescription>
                </Alert>
              )}

              {result && (
                <div className="space-y-6">
                  <div className="grid grid-cols-2 gap-4">
                    <div className="p-4 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-700 dark:to-slate-800 rounded-xl">
                      <div className="flex items-center gap-2 mb-2">
                        <Brain className="h-4 w-4 text-primary" />
                        <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Model Used</span>
                      </div>
                      <Badge variant="secondary" className="text-sm font-semibold">
                        {modelNames[result.model as ModelType] || result.model.toUpperCase()}
                      </Badge>
                    </div>

                    <div className="p-4 bg-gradient-to-br from-slate-50 to-slate-100 dark:from-slate-700 dark:to-slate-800 rounded-xl">
                      <div className="flex items-center gap-2 mb-2">
                        <Clock className="h-4 w-4 text-accent" />
                        <span className="text-sm font-medium text-slate-600 dark:text-slate-400">Processing Time</span>
                      </div>
                      <span className="text-lg font-bold text-slate-900 dark:text-white">
                        {result.inference_ms.toFixed(0)}ms
                      </span>
                    </div>
                  </div>

                  <div className="p-8 rounded-3xl border-2 border-dashed border-primary/30 bg-gradient-to-br from-white via-slate-50/50 to-primary/5 dark:from-slate-800 dark:via-slate-800/50 dark:to-slate-900 relative overflow-hidden">
                    <div className="absolute inset-0 opacity-5">
                      <div className="absolute top-0 left-0 w-32 h-32 bg-primary rounded-full blur-3xl animate-pulse" />
                      <div className="absolute bottom-0 right-0 w-24 h-24 bg-accent rounded-full blur-2xl animate-pulse delay-1000" />
                    </div>

                    <div className="flex items-center gap-4 mb-8 relative z-10">
                      {result.prediction?.toLowerCase().includes("normal") ? (
                        <div className="p-3 bg-gradient-to-br from-green-100 to-green-200 dark:from-green-900/30 dark:to-green-800/30 rounded-2xl shadow-lg">
                          <CheckCircle2 className="h-8 w-8 text-green-600 dark:text-green-400" />
                        </div>
                      ) : (
                        <div className="p-3 bg-gradient-to-br from-red-100 to-red-200 dark:from-red-900/30 dark:to-red-800/30 rounded-2xl shadow-lg">
                          <AlertCircle className="h-8 w-8 text-red-600 dark:text-red-400" />
                        </div>
                      )}
                      <div>
                        <h3 className="text-3xl font-serif font-black text-slate-900 dark:text-white mb-1">
                          {result.prediction}
                        </h3>
                        <p className="text-lg text-slate-600 dark:text-slate-400 font-medium">
                          Confidence: {(Math.max(...result.probs) * 100).toFixed(1)}%
                        </p>
                      </div>
                    </div>

                    <div className="space-y-6 relative z-10">
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-green-700 dark:text-green-400 font-semibold text-lg">Normal</span>
                          <span className="font-black text-xl text-green-700 dark:text-green-400">
                            {(result.probs[0] * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="relative">
                          <Progress
                            value={result.probs[0] * 100}
                            className="h-4 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden"
                          />
                          <div className="absolute inset-0 bg-gradient-to-r from-green-400/20 to-green-600/20 rounded-full animate-pulse" />
                        </div>
                      </div>

                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-red-700 dark:text-red-400 font-semibold text-lg">Pneumonia</span>
                          <span className="font-black text-xl text-red-700 dark:text-red-400">
                            {(result.probs[1] * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="relative">
                          <Progress
                            value={result.probs[1] * 100}
                            className="h-4 bg-slate-200 dark:bg-slate-700 rounded-full overflow-hidden"
                          />
                          <div className="absolute inset-0 bg-gradient-to-r from-red-400/20 to-red-600/20 rounded-full animate-pulse" />
                        </div>
                      </div>
                    </div>
                  </div>

                  <div className="p-4 bg-slate-50 dark:bg-slate-800 rounded-xl">
                    <h4 className="font-medium text-slate-900 dark:text-white mb-2">Image Analysis Details</h4>
                    <div className="grid grid-cols-2 gap-4 text-sm text-slate-600 dark:text-slate-400">
                      <div>
                        Resolution: {result.inferred_shape.H}×{result.inferred_shape.W}px
                      </div>
                      <div>Features: {result.inferred_shape.n_features.toLocaleString()}</div>
                    </div>
                  </div>

                  <Alert className="border-amber-200 bg-amber-50 dark:bg-amber-900/20">
                    <Shield className="h-5 w-5 text-amber-600" />
                    <AlertDescription className="text-base text-amber-800 dark:text-amber-200">
                      This AI prediction is for educational and research purposes only. Always consult with a qualified
                      medical professional for proper diagnosis and treatment decisions.
                    </AlertDescription>
                  </Alert>
                </div>
              )}

              {!result && !error && !isAnalyzing && (
                <div className="text-center py-12">
                  <div className="relative mb-6">
                    <Brain className="h-20 w-20 mx-auto text-slate-300 dark:text-slate-600" />
                    <div className="absolute inset-0 bg-primary/10 rounded-full blur-2xl" />
                  </div>
                  <h3 className="text-xl font-serif font-bold text-slate-900 dark:text-white mb-2">
                    Ready for Analysis
                  </h3>
                  <p className="text-slate-600 dark:text-slate-400">
                    Upload an X-ray image and click "Start Analysis" to begin
                  </p>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  )
}
