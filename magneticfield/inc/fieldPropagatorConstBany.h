// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

// Author: J. Apostolakis  Nov/Dec 2020

#ifndef FIELD_PROPAGATOR_CONST_BANY_H
#define FIELD_PROPAGATOR_CONST_BANY_H

#include <VecGeom/base/Vector3D.h>
#include <CopCore/PhysicalConstants.h>

#include "ConstBzFieldStepper.h"
#include "ConstFieldHelixStepper.h"

using copcore::units::kElectronMassC2;

class fieldPropagatorConstBany {
public:
  __host__ __device__ fieldPropagatorConstBany( vecgeom::Vector3D<double> BfieldVec );
  __host__ __device__ fieldPropagatorConstBany( float BfieldVec[3] );   
  __host__ __device__ ~fieldPropagatorConstBany() {}
   
  __host__ __device__ void stepInField(track                     & aTrack,
                                       // ConstFieldHelixStepper    & helixAnyB,
                                       vecgeom::Vector3D<double> & endPosition,
                                       vecgeom::Vector3D<double> & endDirection);

  __device__ // __host__
  double ComputeStepAndPropagatedState(track &aTrack, float physicsStep, bool& doneFullStep );

  // Auxiliary methods to create / destroy 'clone' objects on device
  __host__ fieldPropagatorConstBany* cloneToDevice();
  __host__ void                      freeClone();
private:

   vecgeom::Vector3D<double> fBfieldVec;
   double                    fBmagnitude;  // field magnitude

   ConstFieldHelixStepper    helixBany;
};

__host__ __device__ fieldPropagatorConstBany::fieldPropagatorConstBany( vecgeom::Vector3D<double> BfieldVec )
   :  helixBany(fBfieldVec)
{
  fBfieldVec  = BfieldVec;
  fBmagnitude = BfieldVec.Mag();
}

__host__ __device__ fieldPropagatorConstBany::fieldPropagatorConstBany( float Bfield[3] )
   :  helixBany( vecgeom::Vector3D<double>( Bfield[0], Bfield[1], Bfield[2] )  )
{
  fBfieldVec  = vecgeom::Vector3D<double>( Bfield[0], Bfield[1], Bfield[2] ) ;
  fBmagnitude = fBfieldVec.Mag();
}


__host__ fieldPropagatorConstBany* fieldPropagatorConstBany::cloneToDevice()
{
  fieldPropagatorConstBany* ptrClone= nullptr;
#ifdef __CUDACC__
  const int objSize = sizeof(fieldPropagatorConstBany);
  char *buffer_dev  = nullptr;
  
  cudaError_t allocErr = cudaMalloc(&buffer_dev, objSize);
  COPCORE_CUDA_CHECK(allocErr);
  cudaError_t  copyErr = cudaMemcpy(buffer_dev, (void *)this, objSize, cudaMemcpyHostToDevice );
  COPCORE_CUDA_CHECK(copyErr);
  
  ptrClone= (fieldPropagatorConstBany*) buffer_dev;
#endif
  return ptrClone;
}

__host__ void fieldPropagatorConstBany::freeClone()
{
#ifdef __CUDACC__
  cudaError_t freeErr = cudaFree(this);
  COPCORE_CUDA_CHECK(freeErr);
#endif   
}

// Cannot make __global__ method part of class
__global__ void moveInField(adept::BlockData<track>  *trackBlock,
                            fieldPropagatorConstBany *fieldProp );

// ----------------------------------------------------------------------------

__host__ __device__ void fieldPropagatorConstBany::stepInField(track &aTrack,
                                                               // ConstFieldHelixStepper    & helixAnyB,
                                                               vecgeom::Vector3D<double> & endPosition,
                                                               vecgeom::Vector3D<double> & endDirection)
{
  int charge  = aTrack.charge();
  double step = aTrack.interaction_length; // was float

  if (charge != 0) {
    double kinE        = aTrack.energy;
    double momentumMag = sqrt(kinE * (kinE + 2.0 * kElectronMassC2));
    // aTrack.mass() -- when extending with other charged particles

    // For now all particles ( e-, e+, gamma ) can be propagated using this
    //   for gammas  charge = 0 works, and ensures that it goes straight.

    helixBany.DoStep(aTrack.pos, aTrack.dir, (double)charge, momentumMag, step, endPosition, endDirection);
  } else {
    // Also move gammas - for now ..
    endPosition  = aTrack.pos + step * aTrack.dir;
    endDirection = aTrack.dir;
  }
}

// -----------------------------------------------------------------------------

// Constant field any direction
//
// was void fieldPropagatorAnyDir_glob(...)
__global__ void moveInField(adept::BlockData<track>   *trackBlock,
                            fieldPropagatorConstBany  *fieldProp )  // pass by value -- pod 
{
  // template <type T> using Vector3D = vecgeom::Vector3D<T>;
  vecgeom::Vector3D<double> endPosition;
  vecgeom::Vector3D<double> endDirection;

  int maxIndex = trackBlock->GetNused() + trackBlock->GetNholes();

  // ConstFieldHelixStepper helixAnyB( fieldProp->GetFieldVec() );

  // Non-block version:
  //   int pclIdx = blockIdx.x * blockDim.x + threadIdx.x;

  for (int pclIdx = blockIdx.x * blockDim.x + threadIdx.x; pclIdx < maxIndex; pclIdx += blockDim.x * gridDim.x) {
    track &aTrack = (*trackBlock)[pclIdx];

    // check if you are not outside the used block
    if (pclIdx >= maxIndex || aTrack.status == dead) continue;

    fieldProp->stepInField(aTrack, /*helixAnyB,*/ endPosition, endDirection);

    // Update position, direction
    aTrack.pos = endPosition;
    aTrack.dir = endDirection;
  }
}


// Determine the step along curved trajectory for charged particles in a field.
//  ( Same name as as navigator method. )
__device__ // __host__
    double
fieldPropagatorConstBany::ComputeStepAndPropagatedState(track &aTrack, float physicsStep, bool& doneFullStep )
{
  // doneFullStep: flag to signify that field integration finished (found boundary or went full length)
   
  // using vec3d = vecgeom::Vector3D<double>;

  if (aTrack.status != alive) return 0.0;

  double kinE        = aTrack.energy;
  double momentumMag = sqrt(kinE * (kinE + 2.0 * copcore::units::kElectronMassC2));

  double charge = aTrack.charge();
  double curv   = std::fabs(ConstBzFieldStepper::kB2C * charge * fBmagnitude) / (momentumMag + 1.0e-30); // norm for step

  constexpr double gEpsilonDeflect = 1.E-2 * copcore::units::cm;
  // acceptable lateral error from field ~ related to delta_chord sagital distance

  // constexpr double invEpsD= 1.0 / gEpsilonDeflect;

  double safeLength =
      2. * sqrt(gEpsilonDeflect / curv); // max length along curve for deflectionn
                                         // = 2. * sqrt( 1.0 / ( invEpsD * curv) ); // Candidate for fast inv-sqrt

  vecgeom::Vector3D<double> position  = aTrack.pos;
  vecgeom::Vector3D<double> direction = aTrack.dir;

  float stepDone = 0.0;
  double remains = physicsStep;

  constexpr double epsilon_step = 1.0e-7; // Ignore remainder if < e_s * PhysicsStep

  if (aTrack.charge() == 0.0) {
    stepDone = LoopNavigator::ComputeStepAndPropagatedState(position, direction, physicsStep, aTrack.current_state,
                                                            aTrack.next_state);
    position += (stepDone + kPushField) * direction;
    doneFullStep = true;
  } else {
    //  Locate the intersection of the curved trajectory and the boundaries of the current
    //    volume (including daughters).
    //  Most electron tracks are short, limited by physics interactions -- the expected
    //    average value of iterations is small.
    //    ( Measuring iterations to confirm the maximum. )
    constexpr int maxChordIters = 10;
    int chordIters              = 0;
    bool remainingStep = true;  // there remains a part of the step to be done

    do {
      bool fullChord = false;  // taken full (current) chord


      vecgeom::Vector3D<double> endPosition  = position;
      vecgeom::Vector3D<double> endDirection = direction;
      double safeMove                        = min(remains, safeLength);

      // fieldPropagatorConstBz( aTrack, BzValue, endPosition, endDirection ); -- Doesn't work
      helixBany.DoStep(position, direction, charge, momentumMag, safeMove, endPosition, endDirection);

      vecgeom::Vector3D<double> chordVec = endPosition - aTrack.pos;
      double chordLen                    = chordVec.Length();
      vecgeom::Vector3D<double> chordDir = (1.0 / chordLen) * chordVec;

      double move = LoopNavigator::ComputeStepAndPropagatedState(position, chordDir, chordLen, aTrack.current_state,
                                                                 aTrack.next_state);

      fullChord = (move == chordLen);
      if (fullChord) {
        position  = endPosition;
        direction = endDirection;
      } else {
        // Accept the intersection point on the surface.  This means that
        //   the point at the boundary will be on the 'straight'-line chord,
        //   not the curved trajectory.
        // ( This involves a bias -- relevant for muons in trackers.
        //   Currently it's controlled/limited by the acceptable step size ie. 'safeLength' )
        position = position + move * chordDir;

        // Primitive approximation of end direction ...
        double fraction = chordLen > 0 ? move / chordLen : 0.0;
        direction       = direction * (1.0 - fraction) + endDirection * fraction;
        direction       = direction.Unit();
      }
      stepDone += move;
      remains -= move;
      chordIters++;

      remainingStep = (!aTrack.next_state.IsOnBoundary()) && fullChord && (remains > epsilon_step * physicsStep);
      
    } while ( remainingStep && (chordIters < maxChordIters));

    doneFullStep = !remainingStep; 

#ifdef CHORD_STATS
    // This stops it from being usable on __host__
    assert(chordIterStatsPBz_dev != nullptr);
    if ((chordIterStatsPBz_dev != nullptr) && chordIters > chordIterStatsPBz_dev->GetMax()) {
      chordIterStatsPBz_dev->updateMax(chordIters);
    }
    chordIterStatsPBz_dev->addIters(chordIters);
#endif
  }
  // stepDone= physicsStep - remains;

  aTrack.pos = position;
  aTrack.dir = direction;

  return stepDone; // physicsStep-remains;
}

#endif
