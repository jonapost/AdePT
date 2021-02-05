// SPDX-FileCopyrightText: 2020 CERN
// SPDX-License-Identifier: Apache-2.0

#include "track.h"
#include "AdePT/LoopNavigator.h"

#include "transportConstants.h"

template <class fieldPropagator_t, bool BfieldOn = true>
class transportation {
public:
   static __host__ __device__ float transport(track &mytrack, fieldPropagator_t &fieldPropagator, float & physics_step, bool & fullLengthInField );
  // fieldPropagator_t fieldPropagator;  // => would need to be device pointer etc
};

// Check whether particle track intersects boundary within length 'physics_step'
//   If so, identify next
// Updates track parameters to end step, including position, direction (if in field).
// Calculates the next navigation state (if encountering boundary
//
// Description last updated: 2021.01.19

template <class fieldPropagator_t, bool BfieldOn>
__host__ __device__ float transportation<fieldPropagator_t, BfieldOn>::transport(track &mytrack,
                                                                                 fieldPropagator_t &fieldPropagator,
                                                                                 float &physics_step,
                                                                                 bool  &fullLengthInField )   
{
  // return value (if step limited by physics or geometry) not used for the moment
  // now, I know which process wins, so I add the particle to the appropriate queue

  float step = 0.0;
  
  if (!BfieldOn) {
    step = LoopNavigator::ComputeStepAndPropagatedState(mytrack.pos, mytrack.dir, physics_step, mytrack.current_state,
                                                        mytrack.next_state);
    mytrack.pos += (step + kPushLinear) * mytrack.dir;
    fullLengthInField= true; // not relevant 
    mytrack.current_process = kBoundaryLinear;    
  } else {
    bool  finishedIntegration= false;  // has field integration finished (found boundary or went full length)     
    step = fieldPropagator.ComputeStepAndPropagatedState(mytrack, physics_step, finishedIntegration);
    // updated state of 'mytrack'
    
    if (step < physics_step ) {
       // Either found a boundary or reached max iterations of integration
       if( finishedIntegration ) {
         assert( mytrack.next_state.IsOnBoundary() && "Field Propagator returned step<phys -- yet NOT boundary!");
         mytrack.current_process = kBoundaryWithField;
      } else {
         mytrack.current_process = kUnfinishedIntegration;
      }
    }
    fullLengthInField= finishedIntegration;
  }
  // if (step < physics_step) mytrack.current_process = BfieldOn ? -2 : -1;

  return step;
}
