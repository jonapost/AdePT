// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  12 Nov 2020

#include <CopCore/SystemOfUnits.h>
#include <CopCore/PhysicalConstants.h>

#include <AdePT/BlockData.h>

#include "ConstBzFieldStepper.h"

using floatX_t = double;  //  float type for X = position
using floatE_t = double;  //  float type for E = energy  & momentum

struct SimpleTrack {
  int      index{0};
  int      pdg{0};
  floatE_t kineticEnergy{0};
  floatX_t position[3]{0};
  floatX_t direction[3]{0};
  floatX_t stepSize;    // Current step size 
  bool     flag1;
  bool     flag2;
};

template<unsigned int N>
struct FieldPropagationBuffer
{
  int      charge[N];
  floatX_t position[3][N];
  floatE_t momentum[3][N];
  int      index[N];
  bool     active[N];
};

constexpr float ElectronMass = copcore::units::kElectronMassC2;

using copcore::units::kElectronMassC2;

using copcore::units::meter;
using copcore::units::GeV;

constexpr floatX_t  minX = -2.0 * meter, maxX = 2.0 * meter;
constexpr floatX_t  minY = -3.0 * meter, maxY = 3.0 * meter;
constexpr floatX_t  minZ = -5.0 * meter, maxZ = 5.0 * meter;

constexpr floatE_t  maxP = 1.0 * GeV;

#include <curand.h>
#include <curand_kernel.h>

__device__ void initOneTrack(int            index,
                             SimpleTrack   &track,
                             curandState_t *states
   )
{
  float r = curand_uniform(states);  
  // track.charge = ( r < 0.45 ? -1 : ( r< 0.9 ? 0 : +1 ) );
  constexpr  int  pdgElec = 11 , pdgGamma = 22;
  track.pdg = ( r < 0.45 ? pdgElec : ( r< 0.9 ? pdgGamma : -pdgElec ) );

  track.position[0] = minX + curand_uniform(states) * ( maxX - minX );
  track.position[1] = minY + curand_uniform(states) * ( maxY - minY );
  track.position[2] = minZ + curand_uniform(states) * ( maxZ - minZ );

  floatE_t  px, py, pz;
  px = maxP * 2.0 * ( curand_uniform(states) - 0.5 );   // -maxP to +maxP
  py = maxP * 2.0 * ( curand_uniform(states) - 0.5 );
  pz = maxP * 2.0 * ( curand_uniform(states) - 0.5 );

  floatE_t  pmag2 =  px*px + py*py + pz*pz;
  floatE_t  inv_pmag = 1.0 * sqrt(pmag2);
  track.direction[0] = px * inv_pmag; 
  track.direction[1] = py * inv_pmag; 
  track.direction[2] = pz * inv_pmag;

  constexpr floatX_t maxStepSize = 0.25 * ( (maxX - minX) + (maxY - minY) + (maxZ - minZ) );
  
  track.stepSize = curand_uniform(states) * maxStepSize;
  
  floatE_t  mass = ( track.pdg == pdgGamma ) ?  0.0 : kElectronMassC2 ; // rest mass
  track.kineticEnergy = pmag2 / ( sqrt( mass * mass + pmag2 ) + mass);
}

// this GPU kernel function is used to initialize 
//     .. the particles' state ?

__global__ void initTracks(adept::BlockData<SimpleTrack> *trackBlock,
                           curandState_t *states,
                           int maxIndex
                          )
{
  /* initialize the tracks with random particles */
  int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;
  if (pclIdx >= maxIndex) return;

  SimpleTrack* pTrack =   trackBlock->NextElement();

  initOneTrack( pclIdx, *pTrack, states);
}

__global__ void initCurand(curandState_t *states)
{
  /* initialize the state */
  curand_init(0, 0, 0, states);
}


constexpr float BzValue = 0.1 * copcore::units::tesla; 

// VECCORE_ATT_HOST_DEVICE
__host__  __device__ 
void EvaluateField( const floatX_t position[3], float fieldValue[3] )
{
    fieldValue[0]= 0.0;
    fieldValue[1]= 0.0;
    fieldValue[2]= BzValue;        
}

// V1 -- one per warp
__global__ void moveInField(adept::BlockData<SimpleTrack> *trackBlock,
                            int maxIndex)
{
  int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;
  SimpleTrack &track= (*trackBlock)[pclIdx];

  // check if you are not outside the used block
  if (pclIdx >= maxIndex ) return;  // || !track.active ) return;
 
  floatX_t  step= track.stepSize;

  // Charge for e+ / e-  only    ( gamma / other neutrals also ok.) 
  int    charge = (track.pdg == -11) - (track.pdg == 11);
  // double pclPosition[3]   
  // Vector3D<double> pclPosition( track.position[0], track.position[1], track.position[2] );

  // Evaluate initial field value
  // EvaluateField( pclPosition, fieldVector );

  // float restMass = ElectronMass;  // For now ... 
  float kinE = track.kineticEnergy;
  floatE_t momentumMag = sqrt( kinE * ( kinE + 2.0 * ElectronMass) );
  
  // Collect position, momentum
  // float momentum[3] = { momentumMag * track.direction[0], 
  //                       momentumMag * track.direction[1], 
  //                       momentumMag * track.direction[2] } ;
  // Vector3D<float> dir( track.direction[0], 
  //                      track.direction[1], 
  //                      track.direction[2] ); 

  floatX_t xOut, yOut, zOut, dirX, dirY, dirZ;

  ConstBzFieldStepper  helixBz(BzValue);

  // For now all particles ( e-, e+, gamma ) can be propagated using this
  //   for gammas  charge = 0 works, and ensures that it goes straight.
  helixBz.DoStep( track.position[0], track.position[1], track.position[2],
                  track.direction[0], track.direction[1], track.direction[2],
                  charge, momentumMag, step,
                  xOut, yOut, zOut, dirX, dirY, dirZ );
  // Alternative: load into local variables ?
  // float xIn= track.position[0], yIn= track.position[1], zIn = track.position[2];
  // float dirXin= track.direction[0], dirYin = track.direction[1], dirZin = track.direction[2];

  // helixBz.DoStep( ); 

  // Update position, direction
  track.position[0] = xOut;
  track.position[1] = yOut;
  track.position[2] = zOut;
  track.direction[0] = dirX;
  track.direction[1] = dirY;
  track.direction[2] = dirZ;
}

int main()
{
  constexpr int numBlocks=1, numThreads=1;
  int  numTracks = 2;
  
  // Initialize Curand
  curandState_t *randState;

  cudaMalloc((void **)&randState, sizeof(curandState_t));
  initCurand<<<numBlocks, numThreads>>>(randState);
  cudaDeviceSynchronize();
  
  // Track capacity of the block
  constexpr int capacity = 1 << 20;

  // Allocate a block of tracks with capacity larger than the total number of spawned threads
  // Note that if we want to allocate several consecutive block in a buffer, we have to use
  // Block_t::SizeOfAlignAware rather than SizeOfInstance to get the space needed per block
  using Block_t    = adept::BlockData<SimpleTrack>;
  size_t blocksize = Block_t::SizeOfInstance(capacity);
  char *buffer2    = nullptr;
  cudaMallocManaged(&buffer2, blocksize);
  auto trackBlock = Block_t::MakeInstanceAt(capacity, buffer2);

  initTracks<<<numBlocks, numThreads>>>(trackBlock, randState, numTracks);
  cudaDeviceSynchronize();

  moveInField<<<numBlocks, numThreads>>>(trackBlock, numTracks);

  cudaDeviceSynchronize();
  // See where they went ?

  constexpr unsigned int SmallNum= 2;
  SimpleTrack tracksHost[SmallNum];
  cudaMemcpy(tracksHost, trackBlock, SmallNum*sizeof(SimpleTrack), cudaMemcpyDeviceToHost );

  for( int i = 0; i<SmallNum ; i++)
     std::cout << " Track " << i << " arrived at x,y,z = " << tracksHost[i].position[0] << " , " << tracksHost[i].position[1]
               << " , " << tracksHost[i].position[3] << std::endl;
  // delete[] tracksHost;
}

