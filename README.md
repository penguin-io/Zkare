# PenguLLM: Privacy-Preserving Personalized Advice with Zero-Knowledge Proofs and LLMs

A cutting-edge implementation of the research paper "Generating Privacy-Preserving Personalized Advice with Zero-Knowledge Proofs and LLMs" that combines zkVM technology with large language models to provide personalized advice while protecting user privacy.

## 🔬 Research Background

This project implements the framework described in the paper that addresses the critical challenge of providing personalized LLM-based advice without compromising user privacy. The system uses Zero-Knowledge Proofs (ZKP) via zkVM to verify user traits without revealing sensitive information.

### Key Innovation
- **Privacy-First Personalization**: Generate personalized advice without exposing sensitive user data
- **zkVM Integration**: Uses RiscZero zkVM for practical zero-knowledge proof generation
- **Two-Entity Architecture**: Separates data holders from advice providers for enhanced privacy
- **Advanced Prompting Strategy**: Leverages both verifiable and unverifiable user traits

## 🏗️ Architecture

```
┌─────────────────┐    ZK Proof +     ┌─────────────────┐
│   Entity 1      │    User Traits    │   Entity 2      │
│                 │ ──────────────────▶│                 │
│ • User Data     │                   │ • LLM Advisor   │
│ • ZK Proof Gen  │                   │ • Advice Gen    │
│ • Trait Inference│                   │ • Proof Verify  │
└─────────────────┘                   └─────────────────┘
      zkVM                                   Llama
```

### Components

1. **Entity 1 (zkproof-service)**: 
   - Holds user data securely
   - Generates zero-knowledge proofs of user traits
   - Implements rule-based inference logic (e.g., financial risk assessment)

2. **Entity 2 (llm-advisor)**: 
   - Provides LLM-based advice using Llama
   - Verifies ZK proofs without accessing raw user data
   - Implements advanced prompting strategies

3. **Web Interface**: 
   - User-friendly interface for interacting with the system
   - Demonstrates the privacy-preserving advice generation

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- NVIDIA GPU (A100 recommended) with CUDA support
- NVIDIA Container Runtime
- At least 32GB RAM (64GB+ recommended)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd PenguLLM
```

2. **Download Llama model** (you'll need access to Llama weights):
```bash
mkdir -p models
# Place your Llama model files in ./models/
# Example structure:
# models/
# ├── llama-3.1-70b/
# │   ├── tokenizer.model
# │   ├── consolidated.*.pth
# │   └── params.json
```

3. **Start the system**:
```bash
docker-compose up -d
```

4. **Access the web interface**:
```
http://localhost:3000
```

### API Endpoints

#### Entity 1 (ZK Proof Service) - Port 8001
- `POST /generate-proof` - Generate ZK proof for user traits
- `GET /verify-proof` - Verify a generated proof
- `POST /infer-traits` - Infer user traits from raw data

#### Entity 2 (LLM Advisor) - Port 8002
- `POST /advice` - Get personalized advice with ZK proof
- `POST /chat` - Interactive chat with privacy-preserving context
- `GET /health` - Service health check

## 🔧 Configuration

### Environment Variables

#### zkproof-service
```env
RUST_LOG=info                 # Logging level
RISC0_DEV_MODE=0             # Production mode for RiscZero
POSTGRES_URL=postgres://...   # Database connection
REDIS_URL=redis://redis:6379  # Cache connection
```

#### llm-advisor
```env
MODEL_PATH=/app/models              # Path to Llama model
ZKPROOF_SERVICE_URL=http://...      # ZK proof service URL
CUDA_VISIBLE_DEVICES=0              # GPU device selection
MAX_CONTEXT_LENGTH=4096             # Maximum context for LLM
TEMPERATURE=0.7                     # LLM sampling temperature
```

### Hardware Requirements

- **Minimum**: 16 vCPU, 64GB RAM, 1x RTX 3090
- **Recommended**: 32 vCPU, 128GB RAM, 1x A100 (24GB)
- **Optimal**: 64 vCPU, 256GB RAM, 2x A100 (80GB)

### Performance Benchmarks

Based on paper evaluation:

| Configuration | ZK Proof Generation | Verification | LLM Response |
|---------------|-------------------|--------------|--------------|
| CPU Only      | 67.8s             | 0.02s        | 2-5s         |
| A100 GPU      | 1.45s             | 0.02s        | 0.5-2s       |

## 📚 Usage Examples

### 1. Financial Risk Assessment

```python
import requests

# Step 1: Submit user data to Entity 1
user_data = {
    "age": 35,
    "income": 75000,
    "savings": 25000,
    "has_mortgage": True,
    "investment_experience": "intermediate",
    "risk_questions": [1, 2, 1, 3, 2, 1, 2, 3, 1, 2]  # 10 financial questions
}

# Generate ZK proof of risk tolerance
response = requests.post("http://localhost:8001/generate-proof", json={
    "user_data": user_data,
    "inference_type": "financial_risk"
})

proof_data = response.json()
# Returns: {"traits": "moderate_risk", "proof": "0x...", "verification_key": "..."}

# Step 2: Get personalized advice from Entity 2
advice_request = {
    "query": "I want to invest $10,000. What should I do?",
    "verified_traits": proof_data["traits"],
    "proof": proof_data["proof"],
    "verification_key": proof_data["verification_key"]
}

advice_response = requests.post("http://localhost:8002/advice", json=advice_request)
print(advice_response.json()["advice"])
```

### 2. Healthcare Consultation

```python
# Healthcare example with age/medical history verification
medical_data = {
    "age": 45,
    "symptoms": ["headache", "fatigue"],
    "medical_history": ["hypertension"],
    "medications": ["lisinopril"]
}

# Generate proof without revealing exact medical details
proof_response = requests.post("http://localhost:8001/generate-proof", json={
    "user_data": medical_data,
    "inference_type": "health_risk"
})

# Get health advice with verified traits
advice = requests.post("http://localhost:8002/advice", json={
    "query": "Should I be concerned about my symptoms?",
    "domain": "healthcare",
    "verified_traits": proof_response.json()["traits"],
    "proof": proof_response.json()["proof"]
})
```

## 🔬 Technical Implementation

### Zero-Knowledge Proof Generation

The system implements the Japanese Bankers Association risk assessment logic as a ZK circuit:

1. **Input Processing**: Parse user responses in JSON format
2. **Hashing**: Apply SHA-256 to question strings for integrity
3. **Scoring**: Rule-based scoring algorithm
4. **Classification**: Categorize into risk levels (Conservative, Steady Growth, Balanced, Aggressive)
5. **Proof Generation**: Create ZK proof that traits were computed correctly

### Prompting Strategy

Implements the paper's two-phase prompting approach:

```
d₀: Unverifiable exploratory traits (e.g., user preferences)
d₁: Verifiable traits (proven via ZKP)

Context Generation:
c₀: Baseline (no emphasis)
c₁: Emphasize d₀ (unverifiable traits)
c₂: Emphasize d₁ (verifiable traits)
c₃: Moderate emphasis on d₁

Response Generation:
A_prop = LLM(Query, I_prop, c_prop)  # Proposed answer
A_exp = LLM(Query, A_prop, I_exp, c_exp)  # Explanation
```

### Privacy Guarantees

- **Zero-Knowledge**: Raw user data never leaves Entity 1
- **Verifiable Computation**: Entity 2 can verify trait computation without seeing inputs
- **Data Minimization**: Only necessary abstracted traits are shared
- **Cryptographic Security**: Based on proven ZK-STARK/SNARK security assumptions

## 🧪 Testing

Run the test suite:

```bash
# Unit tests for ZK proof generation
cd entity1-zkproof && cargo test

# Integration tests for LLM advisor
cd entity2-llm && python -m pytest tests/

# End-to-end system tests
docker-compose exec zkproof-service ./run_tests.sh
docker-compose exec llm-advisor python test_integration.py
```

## 🎯 Evaluation Metrics

The system includes comprehensive evaluation based on the paper's methodology:

### ZK Performance Metrics
- Proof generation time
- Verification time
- Memory usage
- Proof size

### LLM Quality Metrics
- Consistency between A_prop and A_exp
- Trait emphasis effectiveness
- Response relevance and accuracy

### Privacy Metrics
- Information leakage analysis
- Trait inference accuracy
- User acceptance studies

## 🔒 Security Considerations

1. **Proof Integrity**: All ZK proofs use cryptographically secure parameters
2. **Network Security**: TLS encryption between all services
3. **Data Isolation**: Strict separation between Entity 1 and Entity 2
4. **Audit Logging**: Comprehensive logging without exposing sensitive data
5. **Rate Limiting**: Protection against abuse and DoS attacks

## 📊 Monitoring and Observability

Access monitoring dashboards:
- **Grafana**: http://localhost:3001 (performance metrics)
- **Jaeger**: http://localhost:16686 (distributed tracing)
- **Logs**: `docker-compose logs -f <service-name>`

Key metrics monitored:
- ZK proof generation latency
- LLM inference time
- Memory and GPU utilization
- API request rates and errors
- Privacy compliance metrics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
./scripts/setup-dev.sh

# Run in development mode
docker-compose -f docker-compose.dev.yml up

# Format code
./scripts/format.sh

# Run security audit
./scripts/security-audit.sh
```

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 References

- [Original Paper](https://arxiv.org/html/2502.06425v1): "Generating Privacy-Preserving Personalized Advice with Zero-Knowledge Proofs and LLMs"
- [RiscZero Documentation](https://dev.risczero.com/)
- [Llama Model Documentation](https://ai.meta.com/llama/)

## 🙏 Acknowledgments

- Research by Hiroki Watanabe and Motonobu Uchikoshi at The Japan Research Institute
- RiscZero team for zkVM technology
- Meta AI for Llama models
- Open source community for supporting libraries

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/your-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/discussions)
- **Email**: support@pengu-llm.com

---

*Building the future of privacy-preserving AI, one proof at a time.* 🐧🔐