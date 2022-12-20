# 2. Message Passing Neural Networks

### MPNN이란?
- GNN의 가장 기본이 되는 프레임워크로 node 이웃들의 정보를 이용해 해당 node의 상태를 업데이트하는 형태를 가지는 모든 neural network를 의미함
- 해당 글은 node feature x<sub>v</sub>, edge feature e<sub>vw</sub>를 가지는 undirected graph G를 중점적으로 다룸
- forward pass는 message passing phase와 readout phase, 즉 2개의 phase로 이루어짐
  1) message passing phase  
     
     - 정보를 aggregate하는 message function M<sub>t</sub>, hidden state를 update하는 update function U<sub>t</sub>로 이루어짐  
     - M<sub>t</sub>: node v에 대한 정보를 얻기 위해 정보들을 aggregate하는 message function  
       node v 현재 상태, node v 이웃들(N(v))의 현재 상태, node v와 node의 이웃들을 연결하는 edge들의 정보를 aggregate하여 알아보고 싶은 node의 다음 message를 표현함
     - U<sub>t</sub>: 각 node의 다음 hidden state h<sub>v</sub><sup>t+1</sup>를 얻어진 message m<sub>v</sub><sup>t+1</sup>에 근거해 update하는 function
     
          ![message_passing_phase](https://user-images.githubusercontent.com/120429536/208355404-b9300d5a-d6b3-4b11-84d9-5dc0d9e16c6c.PNG)
     
  2) readout phase
  
      - readout function R: 얻은 node의 hidden state를 활용해 원하는 node의 label 등을 예측, 이는 node state의 permutation에 invariant해야함
      
         ![readout_phase](https://user-images.githubusercontent.com/120429536/208357582-bda0170a-cc26-4ae4-8604-1ba437e37501.PNG)
         

### Convolutional Networks for Learning Molecular Fingerprints (2015)
  - Message Function M(h<sub>v</sub>, h<sub>w</sub>, e<sub>vw</sub>): (h<sub>w</sub>, e<sub>vw</sub>) 
    (,:concatenation)
  - message vector m<sub>v</sub><sup>t+1</sup>: Σh<sub>w</sub><sup>t</sup>,Σe<sub>vw</sub>  
    연결된 node와 edge를 각각 분리하여 합해 edge state와 node state 간의 correlation을 확인하기 어렵다는 단점이 존재함
  - Vertex Update Function U(h<sub>v</sub><sup>t</sup>, m<sub>v</sub><sup>t+1</sup>): σ(H<sub>t</sub><sup>deg(v)</sup>m<sub>v</sub><sup>t+1</sup>)  
    (σ: sigmoid function, deg(v): degree of vertex v, H<sub>t</sub><sup>deg(v)</sup>: vertex degree N일 때, step t에서 학습된 update matrix)
  - Readout Function R: f(Σ softmax(W<sub>t</sub>h<sub>v</sub><sup>t</sup>))  
    (f: neural network, W<sub>t</sub>: 학습된 readout matrix)
    
### Gated Graph Neural Networks(GG-NN) (2016)
  - GGNN은 Back-propagation through time 알고리즘을 사용해 모델 매개 변수를 학습하나, 모든 node에서 반복 기능을 여러 번 실행하는 과정에서 중간 상태를 메모리에 저장해야 하므로  
    큰 그래프에서문제가 될 수 있음
  - Message Function M(h<sub>v</sub><sup>t</sup>, h<sub>w</sub><sup>t</sup>, e<sub>vw</sub>): A<sub>e<sub>vw</sub></sub>h<sub>w</sub><sup>t</sup>
    (A<sub>e<sub>vw</sub></sub>: 학습된 matrix)
  - Update Function U<sub>t</sub>: GRU(h<sub>v</sub><sup>t</sup>, m<sub>v</sub><sup>t+1</sup>)
    (GRU: Gated Recurrent Unit, weight tying(가중치 공유)을 통해 동일한 update function 사용)
  - Readout Function R: Σ σ(i(h<sub>v</sub><sup>(T)</sup>, h<sub>v</sub><sup>0</sup>))⊙(j(h<sub>v</sub><sup>(T)</sup>))
    (i,j: neural network, ⊙: element wise multiplication)

### Interaction Networks (2016)
  - Update Function U: 매 time step마다 적용되는 node level effect를 concatenate한 (h<sub>v</sub>, x<sub>v</sub>, m<sub>v</sub>)을 input으로 하는 neural network
    (x<sub>v</sub>: node v의 외부 영향을 의미하는 external vector를 의미함)
  - Message Function M: (h<sub>v</sub>, h<sub>w</sub>, e<sub>vw</sub>)를 concatenate한 neural network
  - Readout Function R: f(Σ h<sub>v</sub><sup>T</sup>)
    (f: 최종 hidden state인 h<sub>v</sub><sup>T</sup>의 sum을 input으로 하는 neural network)
    
### Molecular Graph Convolutions (2016)
  - 다른 MPNN과는 다르게 message passing phase에서 update되는 edge representation인 e<sub>vw</sub><sup>t</sup> 개념이 소개됨
  - Message Function M(h<sub>v</sub><sup>t</sup>, h<sub>w</sub><sup>t</sup>, e<sub>vw</sub><sup>t</sup>): e<sub>vw</sub><sup>t</sup>
  - Update Function U<sub>t</sub>(h<sub>v</sub><sup>t</sup>, m<sub>v</sub><sup>t+1</sup>): α(W1(α(W0h<sub>v</sub><sup>t</sup>), m<sub>v</sub><sup>t+1</sup>))
    (α: RELU activation function, W1, W0: 학습된 weight matrix)
  - edge state update e<sub>vw</sub><sup>t+1</sup>:  U<sub>t</sub><sup>'</sup>(e<sub>vw</sub><sup>t</sup>, h<sub>v</sub><sup>t</sup>, h<sub>w</sub><sup>t</sup>)
    (W<sub>i</sub>: 학습된 weight matrix)
    
### Deep Tensor Neural Networks (2017)
  - Message Function ![image](https://user-images.githubusercontent.com/120429536/208550513-843225e6-80eb-487a-a05c-68e5cf6365a1.png)
  - Update Function U<sub>t</sub>(h<sub>v</sub><sup>t</sup>, m<sub>v</sub><sup>t+1</sup>): h<sub>v</sub><sup>t</sup> + m<sub>v</sub><sup>t+1</sup>
  - Readout Function R: 단층 hidden neural network를 통해 각 node를 독립적으로 pass하며 합하여 output 도출  
    ![image](https://user-images.githubusercontent.com/120429536/208550902-3019837d-754b-4176-9294-f3db0e5d132a.png)

### Laplacian Based Methods (2016)
  - node 값을 나타내는 벡터가 Laplacian matrix와 곱하면, 하나의 node와 이웃 node 값의 차이를 살펴볼 수 있음
  - 이미지에 사용되던 convolution 연산을 graph에 일반화하기 위해 Convolution Theorem 도입
  - Convolution Theorem이란 Graph domain의 convolution은 Fourier domain의 point-wise multiplication과 같다는 의미를 가짐
    - Fourier Transform: 어떤 형태의 주파수가 signal에 어느 정도로 포함되어 있는 지 알 수 있음
    - Graph Fourier Transform: 어떤 형태의 graph 관계가 signal에 어느 정도로 포함되어 있는지 알 수 있음
  
  #### Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering (ChebNet)
  - Spatial Convolution이 아닌 Spectral Convolution을 사용  
    - Spatial Convolution: 고정된 이웃 node에서만 정보를 받아 node의 정보를 update하지만, graph에서는 node간 message passing을 통해 멀리 연결되어 있는 node의 정보도                 밀려 들어올 수 있도록 하는 것이 필요함  
    - Spectral Convolution: 한 node 내에 혼재된 여러 node의 feature를 여러 개의 요소로 나누어 node의 특징을 더 잘 추출하기 위해 spectral 영역에서 convolution을 수행
  - Message Function M(h<sub>v</sub><sup>t</sup>, h<sub>w</sub><sup>t</sup>): C<sub>vw</sub><sup>t</sup>h<sub>w</sub><sup>t</sup>
    (C: graph laplacian L의 eigenvector에 의해 매개변수화되었으며 모델의 학습된 parameter)
  - Update Function U<sub>v</sub><sup>t</sup>(h<sub>v</sub><sup>t</sup>, m<sub>v</sub><sup>t+1</sup>): σ(m<sub>v</sub><sup>t+1</sup>)
    (σ: ReLU와 같은 non-linear activation function)
    
  #### Semi-Supervised classfication with graph convolutional networks (GCN)
  - central node에서 관계가 1만큼 차이가 나는 이웃 node간의 관계를 담은 후, 해당 과정을 layer로 표현한 것이 GCN
  - Message Function M<sub>t</sub>(h<sub>v</sub><sup>t</sup>, h<sub>w</sub><sup>t</sup>): c<sub>vw</sub>h<sub>w</sub><sup>t</sup>
  - 이 때, c<sub>vw</sub> = (deg(v)deg(w))<sup>−1/2</sup>A<sub>vw</sub>이며 이는 서로 노드가 연결되어 있거나 행과 열이 같을 때에만 값이 있고 나머지는 0인 행렬로 이를 통해  
    각 node는 이웃한 node와 본인 node의 hidden layer 값만을 이용해 계속 Update함  
    (CNN이 local feature를 사용하는 특성, weight sharing 특성과 비슷)
  - Update Function U<sub>v</sub><sup>t</sup>(h<sub>v</sub><sup>t</sup>, m<sub>v</sub><sup>t+1</sup>): ReLU(W<sup>t</sup>m<sub>v</sub><sup>t+1</sup>)
