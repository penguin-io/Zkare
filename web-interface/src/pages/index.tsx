import React, { useState, useEffect } from 'react';
import Head from 'next/head';
import { motion } from 'framer-motion';
import {
  ShieldCheckIcon,
  CpuChipIcon,
  SparklesIcon,
  ArrowRightIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon
} from '@heroicons/react/24/outline';
import axios from 'axios';
import toast, { Toaster } from 'react-hot-toast';

interface VerifiedTraits {
  risk_category?: string;
  confidence_score?: number;
  age_bracket?: string;
  income_level?: string;
  experience_level?: string;
}

interface UserProfile {
  age: number;
  income: number;
  savings: number;
  has_mortgage: boolean;
  has_retirement_plan: boolean;
  investment_experience: string;
  risk_questions: number[];
}

interface ProofResponse {
  proof_id: string;
  verified_traits: VerifiedTraits;
  proof_data: string;
  verification_key: string;
  generated_at: string;
  expires_at: string;
}

interface AdviceResponse {
  advice: string;
  explanation: string;
  confidence_score: number;
  domain: string;
  response_time_ms: number;
  used_verified_traits: boolean;
  proof_verified: boolean;
}

export default function Home() {
  const [step, setStep] = useState<'profile' | 'proof' | 'advice'>('profile');
  const [userProfile, setUserProfile] = useState<UserProfile>({
    age: 35,
    income: 75000,
    savings: 25000,
    has_mortgage: true,
    has_retirement_plan: false,
    investment_experience: 'intermediate',
    risk_questions: [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]
  });
  const [proofData, setProofData] = useState<ProofResponse | null>(null);
  const [advice, setAdvice] = useState<AdviceResponse | null>(null);
  const [query, setQuery] = useState('I want to invest $10,000. What should I do?');
  const [loading, setLoading] = useState(false);
  const [services, setServices] = useState({
    zkproof: false,
    llm: false
  });

  // Check service health on component mount
  useEffect(() => {
    checkServiceHealth();
  }, []);

  const checkServiceHealth = async () => {
    try {
      const [zkproofHealth, llmHealth] = await Promise.allSettled([
        axios.get(`${process.env.NEXT_PUBLIC_ZK_SERVICE_URL || 'http://localhost:8001'}/health`),
        axios.get(`${process.env.NEXT_PUBLIC_LLM_SERVICE_URL || 'http://localhost:8002'}/health`)
      ]);

      setServices({
        zkproof: zkproofHealth.status === 'fulfilled' && zkproofHealth.value.status === 200,
        llm: llmHealth.status === 'fulfilled' && llmHealth.value.status === 200
      });
    } catch (error) {
      console.error('Health check failed:', error);
    }
  };

  const generateProof = async () => {
    setLoading(true);
    try {
      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_ZK_SERVICE_URL || 'http://localhost:8001'}/generate-proof`,
        {
          user_data: userProfile,
          inference_type: 'FinancialRisk'
        }
      );

      setProofData(response.data);
      setStep('proof');
      toast.success('Zero-knowledge proof generated successfully!');
    } catch (error) {
      console.error('Proof generation failed:', error);
      toast.error('Failed to generate proof. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const getAdvice = async () => {
    if (!proofData) return;

    setLoading(true);
    try {
      const response = await axios.post(
        `${process.env.NEXT_PUBLIC_LLM_SERVICE_URL || 'http://localhost:8002'}/advice`,
        {
          query,
          domain: 'financial',
          verified_traits: proofData.verified_traits,
          proof: proofData.proof_data,
          verification_key: proofData.verification_key
        }
      );

      setAdvice(response.data);
      setStep('advice');
      toast.success('Personalized advice generated!');
    } catch (error) {
      console.error('Advice generation failed:', error);
      toast.error('Failed to generate advice. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const resetDemo = () => {
    setStep('profile');
    setProofData(null);
    setAdvice(null);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-purple-50">
      <Head>
        <title>PenguLLM - Privacy-Preserving Personalized Advice</title>
        <meta name="description" content="Privacy-Preserving Personalized Advice with Zero-Knowledge Proofs and LLMs" />
        <link rel="icon" href="/favicon.ico" />
      </Head>

      <Toaster position="top-right" />

      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center">
              <div className="w-10 h-10 bg-gradient-to-r from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">🐧</span>
              </div>
              <div className="ml-3">
                <h1 className="text-2xl font-bold text-gray-900">PenguLLM</h1>
                <p className="text-sm text-gray-500">Privacy-Preserving AI Advisor</p>
              </div>
            </div>

            {/* Service Status */}
            <div className="flex space-x-4">
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${services.zkproof ? 'bg-green-400' : 'bg-red-400'}`}></div>
                <span className="text-sm text-gray-600">ZK Proof Service</span>
              </div>
              <div className="flex items-center space-x-2">
                <div className={`w-3 h-3 rounded-full ${services.llm ? 'bg-green-400' : 'bg-red-400'}`}></div>
                <span className="text-sm text-gray-600">LLM Service</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Hero Section */}
        <div className="text-center mb-12">
          <motion.h1
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="text-4xl font-bold text-gray-900 mb-4"
          >
            Privacy-First Personalized Advice
          </motion.h1>
          <motion.p
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="text-xl text-gray-600 mb-8 max-w-3xl mx-auto"
          >
            Experience the future of personalized AI advice with zero-knowledge proofs.
            Get tailored recommendations without compromising your privacy.
          </motion.p>

          {/* Feature Icons */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="flex justify-center space-x-8 mb-8"
          >
            <div className="flex flex-col items-center">
              <ShieldCheckIcon className="w-12 h-12 text-blue-500 mb-2" />
              <span className="text-sm font-medium text-gray-700">Privacy Protected</span>
            </div>
            <div className="flex flex-col items-center">
              <CpuChipIcon className="w-12 h-12 text-purple-500 mb-2" />
              <span className="text-sm font-medium text-gray-700">Zero-Knowledge</span>
            </div>
            <div className="flex flex-col items-center">
              <SparklesIcon className="w-12 h-12 text-green-500 mb-2" />
              <span className="text-sm font-medium text-gray-700">AI Powered</span>
            </div>
          </motion.div>
        </div>

        {/* Progress Steps */}
        <div className="flex justify-center mb-8">
          <div className="flex items-center space-x-4">
            {['profile', 'proof', 'advice'].map((stepName, index) => (
              <div key={stepName} className="flex items-center">
                <div className={`
                  w-10 h-10 rounded-full flex items-center justify-center border-2
                  ${step === stepName || (index === 0 && step === 'profile') ||
                    (index === 1 && (step === 'proof' || step === 'advice')) ||
                    (index === 2 && step === 'advice')
                    ? 'bg-blue-500 border-blue-500 text-white'
                    : 'border-gray-300 text-gray-400'}
                `}>
                  {index + 1}
                </div>
                <span className={`ml-2 text-sm font-medium ${
                  step === stepName ? 'text-blue-600' : 'text-gray-500'
                }`}>
                  {stepName === 'profile' ? 'User Profile' :
                   stepName === 'proof' ? 'ZK Proof' : 'AI Advice'}
                </span>
                {index < 2 && (
                  <ArrowRightIcon className="w-5 h-5 text-gray-400 ml-4" />
                )}
              </div>
            ))}
          </div>
        </div>

        {/* Step Content */}
        <div className="max-w-4xl mx-auto">
          {step === 'profile' && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-lg shadow-lg p-8"
            >
              <h2 className="text-2xl font-bold text-gray-900 mb-6">Financial Profile</h2>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Age</label>
                  <input
                    type="number"
                    value={userProfile.age}
                    onChange={(e) => setUserProfile({...userProfile, age: parseInt(e.target.value)})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Annual Income ($)</label>
                  <input
                    type="number"
                    value={userProfile.income}
                    onChange={(e) => setUserProfile({...userProfile, income: parseInt(e.target.value)})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Savings ($)</label>
                  <input
                    type="number"
                    value={userProfile.savings}
                    onChange={(e) => setUserProfile({...userProfile, savings: parseInt(e.target.value)})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-2">Investment Experience</label>
                  <select
                    value={userProfile.investment_experience}
                    onChange={(e) => setUserProfile({...userProfile, investment_experience: e.target.value})}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="beginner">Beginner</option>
                    <option value="intermediate">Intermediate</option>
                    <option value="advanced">Advanced</option>
                  </select>
                </div>
              </div>

              <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-4">
                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={userProfile.has_mortgage}
                    onChange={(e) => setUserProfile({...userProfile, has_mortgage: e.target.checked})}
                    className="mr-2"
                  />
                  <span className="text-sm text-gray-700">Has Mortgage</span>
                </label>

                <label className="flex items-center">
                  <input
                    type="checkbox"
                    checked={userProfile.has_retirement_plan}
                    onChange={(e) => setUserProfile({...userProfile, has_retirement_plan: e.target.checked})}
                    className="mr-2"
                  />
                  <span className="text-sm text-gray-700">Has Retirement Plan</span>
                </label>
              </div>

              <button
                onClick={generateProof}
                disabled={loading || !services.zkproof}
                className="mt-8 w-full bg-blue-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Generating ZK Proof...' : 'Generate Zero-Knowledge Proof'}
              </button>
            </motion.div>
          )}

          {step === 'proof' && proofData && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-lg shadow-lg p-8"
            >
              <div className="flex items-center mb-6">
                <CheckCircleIcon className="w-8 h-8 text-green-500 mr-3" />
                <h2 className="text-2xl font-bold text-gray-900">Zero-Knowledge Proof Generated</h2>
              </div>

              <div className="bg-gray-50 rounded-lg p-6 mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-4">Verified Traits</h3>
                <div className="space-y-2">
                  <div>
                    <span className="font-medium">Risk Category:</span>
                    <span className="ml-2 capitalize">{proofData.verified_traits.risk_category?.replace('_', ' ')}</span>
                  </div>
                  <div>
                    <span className="font-medium">Age Bracket:</span>
                    <span className="ml-2 capitalize">{proofData.verified_traits.age_bracket?.replace('_', ' ')}</span>
                  </div>
                  <div>
                    <span className="font-medium">Income Level:</span>
                    <span className="ml-2 capitalize">{proofData.verified_traits.income_level?.replace('_', ' ')}</span>
                  </div>
                  <div>
                    <span className="font-medium">Experience Level:</span>
                    <span className="ml-2 capitalize">{proofData.verified_traits.experience_level}</span>
                  </div>
                  <div>
                    <span className="font-medium">Confidence Score:</span>
                    <span className="ml-2">{proofData.verified_traits.confidence_score}%</span>
                  </div>
                </div>
              </div>

              <div className="mb-6">
                <label className="block text-sm font-medium text-gray-700 mb-2">Your Question</label>
                <textarea
                  value={query}
                  onChange={(e) => setQuery(e.target.value)}
                  rows={3}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-blue-500"
                  placeholder="Ask for personalized financial advice..."
                />
              </div>

              <button
                onClick={getAdvice}
                disabled={loading || !services.llm}
                className="w-full bg-purple-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {loading ? 'Generating Advice...' : 'Get Personalized Advice'}
              </button>
            </motion.div>
          )}

          {step === 'advice' && advice && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-white rounded-lg shadow-lg p-8"
            >
              <div className="flex items-center mb-6">
                <SparklesIcon className="w-8 h-8 text-purple-500 mr-3" />
                <h2 className="text-2xl font-bold text-gray-900">Personalized Advice</h2>
              </div>

              <div className="bg-gradient-to-r from-purple-50 to-blue-50 rounded-lg p-6 mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Recommendation</h3>
                <p className="text-gray-700 leading-relaxed">{advice.advice}</p>
              </div>

              <div className="bg-gray-50 rounded-lg p-6 mb-6">
                <h3 className="text-lg font-semibold text-gray-900 mb-3">Explanation</h3>
                <p className="text-gray-700 leading-relaxed">{advice.explanation}</p>
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-6">
                <div className="text-center">
                  <div className="text-2xl font-bold text-blue-600">{Math.round(advice.confidence_score * 100)}%</div>
                  <div className="text-sm text-gray-500">Confidence</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-green-600">{advice.response_time_ms}ms</div>
                  <div className="text-sm text-gray-500">Response Time</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-purple-600">
                    {advice.used_verified_traits ? '✓' : '✗'}
                  </div>
                  <div className="text-sm text-gray-500">Used ZK Traits</div>
                </div>
                <div className="text-center">
                  <div className="text-2xl font-bold text-indigo-600">
                    {advice.proof_verified ? '✓' : '✗'}
                  </div>
                  <div className="text-sm text-gray-500">Proof Verified</div>
                </div>
              </div>

              <button
                onClick={resetDemo}
                className="w-full bg-gray-600 text-white px-6 py-3 rounded-lg font-medium hover:bg-gray-700"
              >
                Try Another Scenario
              </button>
            </motion.div>
          )}
        </div>

        {/* Footer */}
        <footer className="mt-16 text-center text-gray-500">
          <p className="mb-2">
            This demo showcases privacy-preserving personalized advice using Zero-Knowledge Proofs and Large Language Models.
          </p>
          <p className="text-sm">
            Based on the research paper: "Generating Privacy-Preserving Personalized Advice with Zero-Knowledge Proofs and LLMs"
          </p>
        </footer>
      </main>
    </div>
  );
}
