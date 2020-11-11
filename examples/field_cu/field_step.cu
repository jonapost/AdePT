

struct SimpleTrack {
  int index{0};
  int pdg{0};
  double kineticEnergy{0};
  double position[3]{0};
  double direction[3]{0};
  bool flag1;
  bool flag2;
};

template<unsigned int N>
struct WorkBuffer 
{
  float  charge[N];
  float  position[3][N];
  float  momentum[3][N];
  int    index[N];
  bool   active[N];
};

constexpr float ElectronMass = 0.511;  // MeV / c^2

// VECCORE_ATT_HOST_DEVICE
__host__  __device__ 
void EvaluateField( const float position[3], float fieldValue[3] )
{

}



// V1 -- one per warp
__global__ void moveInField(adept::BlockData<SimpleTrack> *block,
                            int maxIndex)
{
  int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;

  // check if you are not outside the used block
  if (pclIdx > maxIndex) return;

  SimpleTrack &track= (*block)[particle_index];
  float pclPosition[3] = { track.position[0], 
                           track.position[1], 
                           track.position[2] };

  EvaluateField( pclPosition, fieldVector );

  // float restMass = ElectronMass;  // For now ... 
  float kinE = track.kineticEnergy;
  float momentumMag = sqrt( kinE * ( kinE + 2.0 * ElectronMass) );
  
  // Collect position, momentum
  float momentum[3] = { momentumMag * track.direction[0], 
                        momentumMag * track.direction[1], 
                        momentumMag * track.direction[2] } ;

  // Evaluate initial field value

}

// V2 -- a work does work of more > 1
//
// kernel function that does transportation
__global__ void transport(int n, adept::BlockData<MyTrack> *block, curandState_t *states, Queue_t *queues)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += blockDim.x * gridDim.x) {
    // transport particles
    for (int xyz = 0; xyz < 3; xyz++) {
      (*block)[i].pos[xyz] = (*block)[i].pos[xyz] + (*block)[i].energy * (*block)[i].dir[xyz];
    }
  }
}
