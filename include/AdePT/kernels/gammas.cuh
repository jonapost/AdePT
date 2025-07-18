// SPDX-FileCopyrightText: 2022 CERN
// SPDX-License-Identifier: Apache-2.0

#include <AdePT/navigation/AdePTNavigator.h>

#include <AdePT/copcore/PhysicalConstants.h>
#include <AdePT/core/TrackDebug.cuh>

#include <G4HepEmGammaManager.hh>
#include <G4HepEmGammaTrack.hh>
#include <G4HepEmTrack.hh>
#include <G4HepEmGammaInteractionCompton.hh>
#include <G4HepEmGammaInteractionConversion.hh>
#include <G4HepEmGammaInteractionPhotoelectric.hh>
// Pull in implementation.
#include <G4HepEmGammaManager.icc>
#include <G4HepEmGammaInteractionCompton.icc>
#include <G4HepEmGammaInteractionConversion.icc>
#include <G4HepEmGammaInteractionPhotoelectric.icc>

using VolAuxData = adeptint::VolAuxData;

#ifdef ASYNC_MODE
namespace AsyncAdePT {
// Asynchronous TransportGammas Interface
template <typename Scoring>
__global__ void __launch_bounds__(256, 1)
    TransportGammas(Track *gammas, const adept::MParray *active, Secondaries secondaries,
                    adept::MParray *nextActiveQueue, adept::MParray *leakedQueue, Scoring *userScoring,
                    Stats *InFlightStats, AllowFinishOffEventArray allowFinishOffEvent, const bool returnAllSteps,
                    const bool returnLastStep)
{
  constexpr Precision kPushDistance = 1000 * vecgeom::kTolerance;
  constexpr unsigned short maxSteps = 10'000;
  int activeSize                    = active->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot      = (*active)[i];
    auto &slotManager   = *secondaries.gammas.fSlotManager;
    Track &currentTrack = gammas[slot];
    auto navState       = currentTrack.navState;

    int lvolID                = navState.GetLogicalId();
    VolAuxData const &auxData = AsyncAdePT::gVolAuxData[lvolID]; // FIXME unify VolAuxData
#else

// Synchronous TransportGammas Interface
template <typename Scoring>
__global__ void TransportGammas(adept::TrackManager<Track> *gammas, Secondaries secondaries, MParrayTracks *leakedQueue,
                                Scoring *userScoring, VolAuxData const *auxDataArray)
{
  using namespace adept_impl;
  constexpr bool returnAllSteps     = false;
  constexpr bool returnLastStep     = false;
  constexpr Precision kPushDistance = 1000 * vecgeom::kTolerance;
  constexpr unsigned short maxSteps = 10'000;
  constexpr int Pdg                 = 22;
  int activeSize                    = gammas->fActiveTracks->size();
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < activeSize; i += blockDim.x * gridDim.x) {
    const int slot = (*gammas->fActiveTracks)[i];
    adeptint::TrackData trackdata;
    Track &currentTrack = (*gammas)[slot];
    auto navState       = currentTrack.navState;

    int lvolID                = navState.GetLogicalId();
    VolAuxData const &auxData = auxDataArray[lvolID]; // FIXME unify VolAuxData

#endif

    bool isLastStep                          = returnLastStep;
    constexpr Precision kPushStuck           = 100 * vecgeom::kTolerance;
    constexpr unsigned short kStepsStuckPush = 5;
    constexpr unsigned short kStepsStuckKill = 25;
    auto eKin                                = currentTrack.eKin;
    auto preStepEnergy                       = eKin;
    auto pos                                 = currentTrack.pos;
    vecgeom::Vector3D<Precision> preStepPos(pos);
    auto dir = currentTrack.dir;
    vecgeom::Vector3D<Precision> preStepDir(dir);
    double globalTime = currentTrack.globalTime;
    double localTime  = currentTrack.localTime;
    double properTime = currentTrack.properTime;
    // the MCC vector is indexed by the logical volume id

    currentTrack.stepCounter++;
    bool printErrors = true;
#if ADEPT_DEBUG_TRACK > 0
    bool verbose = false;
    if (gTrackDebug.active) {
      verbose =
          currentTrack.Matches(gTrackDebug.event_id, gTrackDebug.track_id, gTrackDebug.min_step, gTrackDebug.max_step);
      if (verbose) currentTrack.Print("gamma");
      printErrors = !gTrackDebug.active || verbose;
    }
#endif
    if (currentTrack.stepCounter >= maxSteps || currentTrack.zeroStepCounter > kStepsStuckKill) {
      if (printErrors)
        printf("Killing gamma event %d track %lu E=%f lvol=%d after %d steps with zeroStepCounter %u. This indicates a "
               "stuck particle!\n",
               currentTrack.eventId, currentTrack.trackId, eKin, lvolID, currentTrack.stepCounter,
               currentTrack.zeroStepCounter);
      __syncthreads();
      continue;
    }

    // Write local variables back into track and enqueue
    auto survive = [&](LeakStatus leakReason = LeakStatus::NoLeak) {
      isLastStep = false; // FIXME need to decide on how to do this, as setting the returnLastStep to 0 would delete
                          // the info, but we would need it when the track is leaked. Possible solution: use step
                          // defining process to fix this. returnLastStep && leakReason == LeakStatus::GammaNuclear;
                          // // enable return of last step if it is GammaNuclear
      currentTrack.eKin       = eKin;
      currentTrack.pos        = pos;
      currentTrack.dir        = dir;
      currentTrack.globalTime = globalTime;
      currentTrack.localTime  = localTime;
      currentTrack.properTime = properTime;
      currentTrack.navState   = navState;
      currentTrack.leakStatus = leakReason;
#ifdef ASYNC_MODE
      if (leakReason != LeakStatus::NoLeak) {
        auto success = leakedQueue->push_back(slot);
        if (!success) {
          printf("ERROR: No space left in gammas leaks queue.\n\
\tThe threshold for flushing the leak buffer may be too high\n\
\tThe space allocated to the leak buffer may be too small\n");
          asm("trap;");
        }
      } else {
        nextActiveQueue->push_back(slot);
      }
#else
      currentTrack.CopyTo(trackdata, Pdg);
      if (leakReason != LeakStatus::NoLeak) {
        leakedQueue->push_back(trackdata);
      } else {
        gammas->fNextTracks->push_back(slot);
      }
#endif
    };

#ifdef ASYNC_MODE
    if (InFlightStats->perEventInFlightPrevious[currentTrack.threadId] < allowFinishOffEvent[currentTrack.threadId] &&
        InFlightStats->perEventInFlightPrevious[currentTrack.threadId] != 0) {
      printf("Thread %d Finishing gamma of the %d last particles of event %d on CPU E=%f lvol=%d after %d steps.\n",
             currentTrack.threadId, InFlightStats->perEventInFlightPrevious[currentTrack.threadId],
             currentTrack.eventId, eKin, lvolID, currentTrack.stepCounter);
      survive(LeakStatus::FinishEventOnCPU);
      continue;
    }
#endif

    // Init a track with the needed data to call into G4HepEm.
    G4HepEmGammaTrack gammaTrack;
    G4HepEmTrack *theTrack = gammaTrack.GetTrack();
    theTrack->SetEKin(eKin);
    theTrack->SetMCIndex(auxData.fMCIndex);

    // Re-sample the `number-of-interaction-left` (if needed, otherwise use stored numIALeft) and put it into the
    // G4HepEmTrack. Use index 0 since numIALeft for gammas is based only on the total macroscopic cross section. The
    // currentTrack.numIALeft[0] are updated later
    if (currentTrack.numIALeft[0] <= 0.0) {
      theTrack->SetNumIALeft(-std::log(currentTrack.Uniform()), 0);
    } else {
      theTrack->SetNumIALeft(currentTrack.numIALeft[0], 0);
    }

    // Call G4HepEm to compute the physics step limit.
    G4HepEmGammaManager::HowFar(&g4HepEmData, &g4HepEmPars, &gammaTrack);

    // Get result into variables.
    double geometricalStepLengthFromPhysics = theTrack->GetGStepLength();

#if ADEPT_DEBUG_TRACK > 0
    if (verbose) printf("| geometricalStepLengthFromPhysics %g ", geometricalStepLengthFromPhysics);
#endif

    // Leave the range and MFP inside the G4HepEmTrack. If we split kernels, we
    // also need to carry them over!

    // Check if there's a volume boundary in between.
    vecgeom::NavigationState nextState;
    double geometryStepLength;
#ifdef ADEPT_USE_SURF
    long hitsurf_index = -1;
    geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState,
                                                                  nextState, hitsurf_index);
#else
    geometryStepLength = AdePTNavigator::ComputeStepAndNextVolume(pos, dir, geometricalStepLengthFromPhysics, navState,
                                                                  nextState, kPushDistance);
#endif
    if (geometryStepLength < kPushStuck && geometryStepLength < geometricalStepLengthFromPhysics) {
      currentTrack.zeroStepCounter++;
      if (currentTrack.zeroStepCounter > kStepsStuckPush) geometryStepLength = kPushStuck;
    } else
      currentTrack.zeroStepCounter = 0;

    pos += geometryStepLength * dir;

#if ADEPT_DEBUG_TRACK > 0
    if (verbose) {
      if (currentTrack.zeroStepCounter > kStepsStuckPush) printf("| STUCK TRACK PUSHED ");
      printf("| geometryStepLength %g | propagated_pos {%.19f, %.19f, %.19f} ", geometryStepLength, pos[0], pos[1],
             pos[2]);
      if (geometryStepLength < 100 * vecgeom::kTolerance) printf("| SMALL STEP ");
      nextState.Print();
    }
#endif

    // Set boundary state in navState so the next step and secondaries get the
    // correct information (navState = nextState only if relocated
    // in case of a boundary; see below)
    navState.SetBoundaryState(nextState.IsOnBoundary());

    // Propagate information from geometrical step to G4HepEm.
    theTrack->SetGStepLength(geometryStepLength);
    theTrack->SetOnBoundary(nextState.IsOnBoundary());

    // Update the flight times of the particle
    double deltaTime = theTrack->GetGStepLength() / copcore::units::kCLight;
    globalTime += deltaTime;
    localTime += deltaTime;

    int winnerProcessIndex;
    if (nextState.IsOnBoundary()) {
      // For now, just count that we hit something.

      // Kill the particle if it left the world.
      if (!nextState.IsOutside()) {

        G4HepEmGammaManager::UpdateNumIALeft(theTrack);

        // Save the `number-of-interaction-left` in our track.
        // Use index 0 since numIALeft stores for gammas only the total macroscopic cross section
        double numIALeft          = theTrack->GetNumIALeft(0);
        currentTrack.numIALeft[0] = numIALeft;

#ifdef ADEPT_USE_SURF
        AdePTNavigator::RelocateToNextVolume(pos, dir, hitsurf_index, nextState);
#else
        AdePTNavigator::RelocateToNextVolume(pos, dir, nextState);
#endif

#if ADEPT_DEBUG_TRACK > 0
        if (verbose) {
          printf("| CROSSED into ");
          nextState.Print();
        }
#endif

        // if all steps are returned, we need to record the hit here,
        // as now the nextState is defined, but the navState is not yet replaced
        if (returnAllSteps)
          adept_scoring::RecordHit(userScoring,
                                   currentTrack.trackId,                        // Track ID
                                   currentTrack.parentId,                       // parent Track ID
                                   static_cast<short>(/* transport */ 10),      // step-defining process
                                   2,                                           // Particle type
                                   geometryStepLength,                          // Step length
                                   0,                                           // Total Edep
                                   currentTrack.weight,                         // Track weight
                                   navState,                                    // Pre-step point navstate
                                   preStepPos,                                  // Pre-step point position
                                   preStepDir,                                  // Pre-step point momentum direction
                                   preStepEnergy,                               // Pre-step point kinetic energy
                                   nextState,                                   // Post-step point navstate
                                   pos,                                         // Post-step point position
                                   dir,                                         // Post-step point momentum direction
                                   eKin,                                        // Post-step point kinetic energy
                                   globalTime,                                  // global time
                                   localTime,                                   // local time
                                   currentTrack.eventId, currentTrack.threadId, // event and thread ID
                                   isLastStep,
                                   currentTrack.stepCounter); // whether this is the last step of the track

        // Move to the next boundary.
        navState = nextState;
        // printf("  -> pvol=%d pos={%g, %g, %g} \n", navState.TopId(), pos[0], pos[1], pos[2]);
        //  Check if the next volume belongs to the GPU region and push it to the appropriate queue
        const int nextlvolID = navState.GetLogicalId();

#ifdef ASYNC_MODE // FIXME unify VolAuxData
        VolAuxData const &nextauxData = AsyncAdePT::gVolAuxData[nextlvolID];
#else
        VolAuxData const &nextauxData = auxDataArray[nextlvolID];
#endif

        if (nextauxData.fGPUregion > 0)
          survive();
        else {
          // To be safe, just push a bit the track exiting the GPU region to make sure
          // Geant4 does not relocate it again inside the same region
          pos += kPushDistance * dir;

#if ADEPT_DEBUG_TRACK > 0
          if (verbose) printf("\n| track leaked to Geant4\n");
#endif

          survive(LeakStatus::OutOfGPURegion);
        }
      } else {
        // release slot for particle that has left the world
#ifdef ASYNC_MODE
        slotManager.MarkSlotForFreeing(slot);
#endif

        // particle has left the world, record hit if last or all steps are returned
        if (returnAllSteps || returnLastStep)
          adept_scoring::RecordHit(userScoring,
                                   currentTrack.trackId,                        // Track ID
                                   currentTrack.parentId,                       // parent Track ID
                                   static_cast<short>(/* transport */ 10),      // step-defining process
                                   2,                                           // Particle type
                                   geometryStepLength,                          // Step length
                                   0,                                           // Total Edep
                                   currentTrack.weight,                         // Track weight
                                   navState,                                    // Pre-step point navstate
                                   preStepPos,                                  // Pre-step point position
                                   preStepDir,                                  // Pre-step point momentum direction
                                   preStepEnergy,                               // Pre-step point kinetic energy
                                   nextState,                                   // Post-step point navstate
                                   pos,                                         // Post-step point position
                                   dir,                                         // Post-step point momentum direction
                                   eKin,                                        // Post-step point kinetic energy
                                   globalTime,                                  // global time
                                   localTime,                                   // local time
                                   currentTrack.eventId, currentTrack.threadId, // event and thread ID
                                   isLastStep,                // whether this is the last step of the track
                                   currentTrack.stepCounter); // whether this is the first step
      }
      continue;
    } else {

      G4HepEmGammaManager::SampleInteraction(&g4HepEmData, &gammaTrack, currentTrack.Uniform());
      winnerProcessIndex = theTrack->GetWinnerProcessIndex();

#if ADEPT_DEBUG_TRACK > 0
      if (verbose) printf("| winnerProc %d\n", winnerProcessIndex);
#endif

      // Reset number of interaction left for the winner discrete process also in the currentTrack
      // (SampleInteraction() resets it for theTrack), will be resampled in the next iteration.
      currentTrack.numIALeft[0] = -1.0;
    }

    // Perform the discrete interaction.
    G4HepEmRandomEngine rnge(&currentTrack.rngState);
    // We might need one branched RNG state, prepare while threads are synchronized.
    RanluxppDouble newRNG(currentTrack.rngState.Branch());

    const double theElCut    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecElProdCutE;
    const double thePosCut   = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecPosProdCutE;
    const double theGammaCut = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fSecGamProdCutE;

    const int iregion    = g4HepEmData.fTheMatCutData->fMatCutData[auxData.fMCIndex].fG4RegionIndex;
    const bool ApplyCuts = g4HepEmPars.fParametersPerRegion[iregion].fIsApplyCuts;

    double edep           = 0.;
    double newEnergyGamma = 0.; // gamma energy after compton scattering

    switch (winnerProcessIndex) {
    case 0: {
      // Invoke gamma conversion to e-/e+ pairs, if the energy is above the threshold.
      if (eKin < 2 * copcore::units::kElectronMassC2) {
        survive();
        break;
      }

#if ADEPT_DEBUG_TRACK > 0
      if (verbose) printf("| GAMMA CONVERSION ");
#endif

      double logEnergy = std::log(eKin);
      double elKinEnergy, posKinEnergy;
      G4HepEmGammaInteractionConversion::SampleKinEnergies(&g4HepEmData, eKin, logEnergy, auxData.fMCIndex, elKinEnergy,
                                                           posKinEnergy, &rnge);

      double dirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirSecondaryEl[3], dirSecondaryPos[3];
      G4HepEmGammaInteractionConversion::SampleDirections(dirPrimary, dirSecondaryEl, dirSecondaryPos, elKinEnergy,
                                                          posKinEnergy, &rnge);

      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 1, /*numGammas*/ 0);

      // Check the cuts and deposit energy in this volume if needed
      if (ApplyCuts && elKinEnergy < theElCut) {
        // Deposit the energy here and kill the secondary
        edep = elKinEnergy;
      } else {
#ifdef ASYNC_MODE
        Track &electron = secondaries.electrons.NextTrack(
            newRNG, elKinEnergy, pos,
            vecgeom::Vector3D<Precision>{dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]}, navState,
            currentTrack, globalTime);
#else
        Track &electron = secondaries.electrons->NextTrack();
        electron.InitAsSecondary(pos, navState, globalTime);
        electron.parentId = currentTrack.trackId;
        electron.rngState = newRNG;
        electron.trackId  = electron.rngState.IntRndm64();
        electron.eKin     = elKinEnergy;
        electron.weight   = currentTrack.weight;
        electron.dir.Set(dirSecondaryEl[0], dirSecondaryEl[1], dirSecondaryEl[2]);
#endif
        // if tracking or stepping action is called, return initial step
        if (returnLastStep) {
          adept_scoring::RecordHit(userScoring, electron.trackId, electron.parentId,
                                   /*CreatorProcessId*/ short(winnerProcessIndex),
                                   /* electron*/ 0,                     // Particle type
                                   0,                                   // Step length
                                   0,                                   // Total Edep
                                   electron.weight,                     // Track weight
                                   navState,                            // Pre-step point navstate
                                   electron.pos,                        // Pre-step point position
                                   electron.dir,                        // Pre-step point momentum direction
                                   electron.eKin,                       // Pre-step point kinetic energy
                                   navState,                            // Post-step point navstate
                                   electron.pos,                        // Post-step point position
                                   electron.dir,                        // Post-step point momentum direction
                                   electron.eKin,                       // Post-step point kinetic energy
                                   globalTime,                          // global time
                                   0.,                                  // local time
                                   electron.eventId, electron.threadId, // eventID and threadID
                                   false,                               // whether this was the last step
                                   electron.stepCounter);               // whether this was the first step
        }
      }

      if (ApplyCuts && (copcore::units::kElectronMassC2 < theGammaCut && posKinEnergy < thePosCut)) {
        // Deposit: posKinEnergy + 2 * copcore::units::kElectronMassC2 and kill the secondary
        edep += posKinEnergy + 2 * copcore::units::kElectronMassC2;
      } else {
#ifdef ASYNC_MODE
        Track &positron = secondaries.positrons.NextTrack(
            currentTrack.rngState, posKinEnergy, pos,
            vecgeom::Vector3D<Precision>{dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]}, navState,
            currentTrack, globalTime);
#else
        Track &positron = secondaries.positrons->NextTrack();
        positron.InitAsSecondary(pos, navState, globalTime);
        // Reuse the RNG state of the dying track.
        positron.parentId = currentTrack.trackId;
        positron.rngState = currentTrack.rngState;
        positron.trackId  = positron.rngState.IntRndm64();
        positron.eKin     = posKinEnergy;
        positron.weight   = currentTrack.weight;
        positron.dir.Set(dirSecondaryPos[0], dirSecondaryPos[1], dirSecondaryPos[2]);
#endif
        // if tracking or stepping action is called, return initial step
        if (returnLastStep) {
          adept_scoring::RecordHit(userScoring, positron.trackId, positron.parentId,
                                   /*CreatorProcessId*/ short(winnerProcessIndex),
                                   /* positron*/ 1,                     // Particle type
                                   0,                                   // Step length
                                   0,                                   // Total Edep
                                   positron.weight,                     // Track weight
                                   navState,                            // Pre-step point navstate
                                   positron.pos,                        // Pre-step point position
                                   positron.dir,                        // Pre-step point momentum direction
                                   positron.eKin,                       // Pre-step point kinetic energy
                                   navState,                            // Post-step point navstate
                                   positron.pos,                        // Post-step point position
                                   positron.dir,                        // Post-step point momentum direction
                                   positron.eKin,                       // Post-step point kinetic energy
                                   globalTime,                          // global time
                                   0.,                                  // local time
                                   positron.eventId, positron.threadId, // eventID and threadID
                                   false,                               // whether this was the last step
                                   positron.stepCounter);               // whether this was the first step
        }
      }

      // The current track is killed by not enqueuing into the next activeQueue and the slot is released
#ifdef ASYNC_MODE
      slotManager.MarkSlotForFreeing(slot);
#endif
      break;
    }
    case 1: {
      // Invoke Compton scattering of gamma.

#if ADEPT_DEBUG_TRACK > 0
      if (verbose) printf("| COMPTON ");
#endif

      constexpr double LowEnergyThreshold = 100 * copcore::units::eV;
      if (eKin < LowEnergyThreshold) {
        survive();
        break;
      }
      const double origDirPrimary[] = {dir.x(), dir.y(), dir.z()};
      double dirPrimary[3];
      newEnergyGamma =
          G4HepEmGammaInteractionCompton::SamplePhotonEnergyAndDirection(eKin, dirPrimary, origDirPrimary, &rnge);
      vecgeom::Vector3D<double> newDirGamma(dirPrimary[0], dirPrimary[1], dirPrimary[2]);

      const double energyEl = eKin - newEnergyGamma;

      adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

      // Check the cuts and deposit energy in this volume if needed
      if (ApplyCuts ? energyEl > theElCut : energyEl > LowEnergyThreshold) {
        // Create a secondary electron and sample/compute directions.
#ifdef ASYNC_MODE
        Track &electron = secondaries.electrons.NextTrack(
            newRNG, energyEl, pos, eKin * dir - newEnergyGamma * newDirGamma, navState, currentTrack, globalTime);
#else
        Track &electron = secondaries.electrons->NextTrack();

        electron.InitAsSecondary(pos, navState, globalTime);
        electron.parentId = currentTrack.trackId;
        electron.rngState = newRNG;
        electron.trackId  = electron.rngState.IntRndm64();
        electron.eKin     = energyEl;
        electron.weight   = currentTrack.weight;
        electron.dir      = eKin * dir - newEnergyGamma * newDirGamma;
#endif
        electron.dir.Normalize();

        // if tracking or stepping action is called, return initial step
        if (returnLastStep) {
          adept_scoring::RecordHit(userScoring, electron.trackId, electron.parentId,
                                   /*CreatorProcessId*/ short(winnerProcessIndex),
                                   /* electron*/ 0,                     // Particle type
                                   0,                                   // Step length
                                   0,                                   // Total Edep
                                   electron.weight,                     // Track weight
                                   navState,                            // Pre-step point navstate
                                   electron.pos,                        // Pre-step point position
                                   electron.dir,                        // Pre-step point momentum direction
                                   electron.eKin,                       // Pre-step point kinetic energy
                                   navState,                            // Post-step point navstate
                                   electron.pos,                        // Post-step point position
                                   electron.dir,                        // Post-step point momentum direction
                                   electron.eKin,                       // Post-step point kinetic energy
                                   globalTime,                          // global time
                                   0.,                                  // local time
                                   electron.eventId, electron.threadId, // eventID and threadID
                                   false,                               // whether this was the last step
                                   electron.stepCounter);               // whether this was the first step
        }
      } else {
        edep = energyEl;
      }

      // Check the new gamma energy and deposit if below threshold.
      // Using same hardcoded very LowEnergyThreshold as G4HepEm
      if (newEnergyGamma > LowEnergyThreshold) {
        eKin = newEnergyGamma;
        dir  = newDirGamma;
        survive();
      } else {
        edep += newEnergyGamma;
        newEnergyGamma = 0.;
        // The current track is killed by not enqueuing into the next activeQueue and the slot is released
#ifdef ASYNC_MODE
        slotManager.MarkSlotForFreeing(slot);
#endif
      }
      break;
    }
    case 2: {
      // Invoke photoelectric process.

#if ADEPT_DEBUG_TRACK > 0
      if (verbose) printf("| PHOTOELECTRIC ");
#endif

      const double theLowEnergyThreshold = 1 * copcore::units::eV;

      const double bindingEnergy = G4HepEmGammaInteractionPhotoelectric::SelectElementBindingEnergy(
          &g4HepEmData, auxData.fMCIndex, gammaTrack.GetPEmxSec(), eKin, &rnge);

      edep                    = bindingEnergy;
      const double photoElecE = eKin - edep;
      if (ApplyCuts ? photoElecE > theElCut : photoElecE > theLowEnergyThreshold) {

        adept_scoring::AccountProduced(userScoring, /*numElectrons*/ 1, /*numPositrons*/ 0, /*numGammas*/ 0);

        double dirGamma[] = {dir.x(), dir.y(), dir.z()};
        double dirPhotoElec[3];
        G4HepEmGammaInteractionPhotoelectric::SamplePhotoElectronDirection(photoElecE, dirGamma, dirPhotoElec, &rnge);

        // Create a secondary electron and sample directions.
#ifdef ASYNC_MODE
        Track &electron = secondaries.electrons.NextTrack(
            newRNG, photoElecE, pos, vecgeom::Vector3D<Precision>{dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]},
            navState, currentTrack, globalTime);
#else
        Track &electron = secondaries.electrons->NextTrack();
        electron.InitAsSecondary(pos, navState, globalTime);
        electron.parentId = currentTrack.trackId;
        electron.rngState = newRNG;
        electron.trackId  = electron.rngState.IntRndm64();
        electron.eKin     = photoElecE;
        electron.weight   = currentTrack.weight;
        electron.dir.Set(dirPhotoElec[0], dirPhotoElec[1], dirPhotoElec[2]);
#endif
        // if tracking or stepping action is called, return initial step
        if (returnLastStep) {
          adept_scoring::RecordHit(userScoring, electron.trackId, electron.parentId,
                                   /*CreatorProcessId*/ short(winnerProcessIndex),
                                   /* electron*/ 0,                     // Particle type
                                   0,                                   // Step length
                                   0,                                   // Total Edep
                                   electron.weight,                     // Track weight
                                   navState,                            // Pre-step point navstate
                                   electron.pos,                        // Pre-step point position
                                   electron.dir,                        // Pre-step point momentum direction
                                   electron.eKin,                       // Pre-step point kinetic energy
                                   navState,                            // Post-step point navstate
                                   electron.pos,                        // Post-step point position
                                   electron.dir,                        // Post-step point momentum direction
                                   electron.eKin,                       // Post-step point kinetic energy
                                   globalTime,                          // global time
                                   0.,                                  // local time
                                   electron.eventId, electron.threadId, // eventID and threadID
                                   false,                               // whether this was the last step
                                   electron.stepCounter);               // whether this was the first step
        }
      } else {
        // If the secondary electron is cut, deposit all the energy of the gamma in this volume
        edep = eKin;
      }
      // The current track is killed by not enqueuing into the next activeQueue and the slot is released
#ifdef ASYNC_MODE
      slotManager.MarkSlotForFreeing(slot);
#endif
      break;
    }
    case 3: {

#if ADEPT_DEBUG_TRACK > 0
      if (verbose) printf("| GAMMA-NUCLEAR ");
#endif
      // Gamma nuclear needs to be handled by Geant4 directly, passing track back to CPU
      survive(LeakStatus::GammaNuclear);
    }
    }

    // If there is some edep from cutting particles, record the step
    if ((edep > 0 && auxData.fSensIndex >= 0) || returnAllSteps || returnLastStep) {
      adept_scoring::RecordHit(userScoring,
                               currentTrack.trackId,                        // Track ID
                               currentTrack.parentId,                       // parent Track ID
                               short(winnerProcessIndex),                   // step-defining process id
                               2,                                           // Particle type
                               geometryStepLength,                          // Step length
                               edep,                                        // Total Edep
                               currentTrack.weight,                         // Track weight
                               navState,                                    // Pre-step point navstate
                               preStepPos,                                  // Pre-step point position
                               preStepDir,                                  // Pre-step point momentum direction
                               preStepEnergy,                               // Pre-step point kinetic energy
                               nextState,                                   // Post-step point navstate
                               pos,                                         // Post-step point position
                               dir,                                         // Post-step point momentum direction
                               newEnergyGamma,                              // Post-step point kinetic energy
                               globalTime,                                  // global time
                               localTime,                                   // local time
                               currentTrack.eventId, currentTrack.threadId, // event and thread ID
                               isLastStep,                // whether this is the last step of the track
                               currentTrack.stepCounter); // whether this is the first step
    }
  }
}

#ifdef ASYNC_MODE
} // namespace AsyncAdePT
#endif
