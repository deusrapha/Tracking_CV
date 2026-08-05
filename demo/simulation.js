/**
 * Beyond the Line of Sight: Occlusion Tracking Demonstration
 * JavaScript Simulation Engine, CTRV Kalman Filter, and Amodal Anchor Logic
 */

// --- 1. Math and Matrix Utilities ---
const MatrixUtils = {
    // 3x3 matrix multiplication: A * B
    mult3x3(A, B) {
        let C = new Array(9).fill(0);
        for(let i=0; i<3; ++i) {
            for(let j=0; j<3; ++j) {
                let s = 0;
                for(let k=0; k<3; ++k) {
                    s += A[i*3 + k] * B[k*3 + j];
                }
                C[i*3 + j] = s;
            }
        }
        return C;
    },
    // 3x3 matrix times 3x1 vector
    mult3x3Vec(A, v) {
        return [
            A[0]*v[0] + A[1]*v[1] + A[2]*v[2],
            A[3]*v[0] + A[4]*v[1] + A[5]*v[2],
            A[6]*v[0] + A[7]*v[1] + A[8]*v[2]
        ];
    },
    // Invert 3x3 matrix
    inv3x3(A) {
        let det = A[0] * (A[4]*A[8] - A[5]*A[7]) -
                  A[1] * (A[3]*A[8] - A[5]*A[6]) +
                  A[2] * (A[3]*A[7] - A[4]*A[6]);
        if (Math.abs(det) < 1e-6) return new Array(9).fill(0);
        let invdet = 1.0 / det;
        return [
             (A[4]*A[8] - A[5]*A[7]) * invdet,
            -(A[1]*A[8] - A[2]*A[7]) * invdet,
             (A[1]*A[5] - A[2]*A[4]) * invdet,
            -(A[3]*A[8] - A[5]*A[6]) * invdet,
             (A[0]*A[8] - A[2]*A[6]) * invdet,
            -(A[0]*A[5] - A[2]*A[3]) * invdet,
             (A[3]*A[7] - A[4]*A[6]) * invdet,
            -(A[0]*A[7] - A[1]*A[6]) * invdet,
             (A[0]*A[4] - A[1]*A[3]) * invdet
        ];
    }
};

// --- 2. Ground Plane Projector (Homography Warp) ---
class GroundPlaneProjector {
    constructor(imgW = 1024, imgH = 576) {
        this.imgW = imgW;
        this.imgH = imgH;
        // DJI Camera approximation (focal length, center)
        let f = 800.0;
        let cx = imgW / 2.0;
        let cy = imgH / 2.0;
        this.K = [
            f, 0, cx,
            0, f, cy,
            0, 0, 1
        ];
        this.K_inv = MatrixUtils.inv3x3(this.K);
    }

    computeHomography(altitude, pitchDeg, rollDeg) {
        let pitch = pitchDeg * Math.PI / 180.0;
        let roll = rollDeg * Math.PI / 180.0;

        // Rotation matrix: pitch around X, roll around Y
        let R_pitch = [
            1, 0, 0,
            0, Math.cos(pitch), -Math.sin(pitch),
            0, Math.sin(pitch), Math.cos(pitch)
        ];
        let R_roll = [
            Math.cos(roll), 0, Math.sin(roll),
            0, 1, 0,
            -Math.sin(roll), 0, Math.cos(roll)
        ];
        
        // R = R_pitch @ R_roll
        let R = MatrixUtils.mult3x3(R_pitch, R_roll);

        // Columns of R
        let R1 = [R[0], R[3], R[6]];
        let R2 = [R[1], R[4], R[7]];
        let t  = [0, 0, altitude]; // UAV position in world space

        // H_g2i = K @ [R1, R2, t] (3x3 matrix mapping ground to image)
        let g2i = [
            this.K[0]*R1[0] + this.K[1]*R1[1] + this.K[2]*R1[2], this.K[0]*R2[0] + this.K[1]*R2[1] + this.K[2]*R2[2], this.K[0]*t[0] + this.K[1]*t[1] + this.K[2]*t[2],
            this.K[3]*R1[0] + this.K[4]*R1[1] + this.K[5]*R1[2], this.K[3]*R2[0] + this.K[4]*R2[1] + this.K[5]*R2[2], this.K[3]*t[0] + this.K[4]*t[1] + this.K[5]*t[2],
            this.K[6]*R1[0] + this.K[7]*R1[1] + this.K[8]*R1[2], this.K[6]*R2[0] + this.K[7]*R2[1] + this.K[8]*R2[2], this.K[6]*t[0] + this.K[7]*t[1] + this.K[8]*t[2]
        ];

        // We need image to ground: H_i2g = inverse(H_g2i)
        let i2g = MatrixUtils.inv3x3(g2i);
        return { g2i, i2g };
    }

    projectImageToGround(u, v, H_i2g) {
        let g = MatrixUtils.mult3x3Vec(H_i2g, [u, v, 1.0]);
        if (Math.abs(g[2]) > 1e-5) {
            return [g[0] / g[2], g[1] / g[2]];
        }
        return [g[0], g[1]];
    }

    projectGroundToImage(x, y, H_g2i) {
        let img = MatrixUtils.mult3x3Vec(H_g2i, [x, y, 1.0]);
        if (Math.abs(img[2]) > 1e-5) {
            return [img[0] / img[2], img[1] / img[2]];
        }
        return [img[0], img[1]];
    }
}

/// --- 3. Unified Counterfactual Amodal Track (CAT) ---
class CounterfactualAmodalTrack {
    constructor(bbox, trackId, appearanceHist) {
        this.trackId = trackId;
        
        let cx = bbox[0] + (bbox[2] - bbox[0]) / 2.0;
        let cy = bbox[1] + (bbox[3] - bbox[1]) / 2.0;
        let w = bbox[2] - bbox[0];
        let h = bbox[3] - bbox[1];

        // 1. TrackState
        this.state = "NEW"; // "NEW", "VISIBLE", "OCCLUDED", "REMERGING", "LOST", "SEARCH", "EXPIRED"

        // 2. MotionState (7-state EKF CTRV model)
        this.motion = {
            cx: cx,
            cy: cy,
            v: 0.0,
            theta: 0.0,
            omega: 0.0,
            w: w,
            h: h,
            // Covariances
            P: [
                10, 0, 0, 0, 0, 0, 0,
                0, 10, 0, 0, 0, 0, 0,
                0, 0, 5, 0, 0, 0, 0,
                0, 0, 0, 1, 0, 0, 0,
                0, 0, 0, 0, 0.1, 0, 0,
                0, 0, 0, 0, 0, 10, 0,
                0, 0, 0, 0, 0, 0, 10
            ],
            Q_diag: [0.1, 0.1, 0.2, 0.1, 0.05, 0.1, 0.1],
            R_diag: [2.0, 2.0, 4.0, 4.0],
            // Growing 2D uncertainty for amodal gating
            Sigma: [5.0, 0.0, 0.0, 5.0],
            Q_sigma: [3.0, 0.0, 0.0, 3.0]
        };

        // 3. IdentityState
        this.identity = {
            appearanceEmbedding: appearanceHist,
            confidence: 1.0,
            age: 0,
            lastMatchedFrame: 0,
            verificationScore: 1.0
        };

        // 4. SocialState
        this.social = {
            blendFactor: 0.0
        };

        // 4b. BehaviorState
        this.behavior = {
            mode: "walking",
            headingHistory: [],      // sliding window of last 8 headings
            speedHistory: [],        // sliding window of last 8 speeds
            turnPersistence: 0.0,
            isLeader: false
        };
        this.counterfactualConfidence = 1.0;

        // 5. OcclusionState
        this.occlusion = {
            framesOccluded: 0,
            lastVelocity: [0.0, 0.0],
            entryFrame: 0
        };

        // 6. MemoryState
        this.memory = {
            trajectoryHistory: [], // array of { cx, cy, state }
            velocityHistory: []
        };
        this.lastUpdatedCx = cx;
        this.lastUpdatedCy = cy;
        this.identityMemory = new IdentityMemory(appearanceHist, appearanceHist, w / h, 0.0, 0.0, 1.0);
    }

    predict(dt = 1.0, herdMeanVelocity = null, cohesionWeight = 0.5) {
        // Increment general age
        this.identity.age += dt;

        // CTRV kinematics update
        let cx = this.motion.cx;
        let cy = this.motion.cy;
        let v = this.motion.v;
        let theta = this.motion.theta;
        let omega = this.motion.omega;
        let w = this.motion.w;
        let h = this.motion.h;

        // Classify current behavior state mode
        let qScale = 0.5;
        if (v < 0.2) {
            this.behavior.mode = "stationary";
            qScale = 0.05;
        } else if (Math.abs(omega) > 0.15) {
            this.behavior.mode = "sharp turning";
            qScale = 2.0;
        } else if (v > 3.0) {
            this.behavior.mode = "running";
            qScale = 1.2;
        } else {
            this.behavior.mode = "walking";
            qScale = 0.5;
        }

        // Perform prediction depending on state
        if (this.state === "OCCLUDED" || this.state === "REMERGING" || this.state === "SEARCH" || this.state === "LOST") {
            this.occlusion.framesOccluded += dt;

            // Propagate heading using EKF turn rate omega
            theta += omega * dt;
            this.motion.theta = theta;

            // Velocity blending with herd social prior
            let selfVx = v * Math.cos(theta);
            let selfVy = v * Math.sin(theta);
            let blendedVx = selfVx;
            let blendedVy = selfVy;
            if (herdMeanVelocity) {
                blendedVx = (1.0 - cohesionWeight) * selfVx + cohesionWeight * herdMeanVelocity[0];
                blendedVy = (1.0 - cohesionWeight) * selfVy + cohesionWeight * herdMeanVelocity[1];
            }

            // Propagate amodal center
            this.motion.cx += blendedVx * dt;
            this.motion.cy += blendedVy * dt;

            // Grow spatial amodal uncertainty Sigma
            this.motion.Sigma[0] += this.motion.Q_sigma[0] * qScale * dt;
            this.motion.Sigma[1] += this.motion.Q_sigma[1] * qScale * dt;
            this.motion.Sigma[2] += this.motion.Q_sigma[2] * qScale * dt;
            this.motion.Sigma[3] += this.motion.Q_sigma[3] * qScale * dt;

            // Propagate covariance P under occlusion
            for (let i = 0; i < 7; i++) {
                this.motion.P[i*7 + i] += this.motion.Q_diag[i] * qScale * dt;
            }

            // Calculate dynamic counterfactual confidence score
            let sigmaTrace = this.motion.Sigma[0] + this.motion.Sigma[3];
            let motionConf = Math.exp(-0.005 * Math.max(0.0, sigmaTrace - 10.0));
            let appearanceConf = Math.pow(0.995, this.occlusion.framesOccluded);
            
            let herdConf = 1.0;
            if (herdMeanVelocity) {
                let normSelf = Math.hypot(selfVx, selfVy);
                let normHerd = Math.hypot(herdMeanVelocity[0], herdMeanVelocity[1]);
                if (normSelf > 0.1 && normHerd > 0.1) {
                    let cosSim = (selfVx * herdMeanVelocity[0] + selfVy * herdMeanVelocity[1]) / (normSelf * normHerd);
                    herdConf = 0.5 + 0.5 * Math.max(0.0, cosSim);
                }
            }
            let terrainConf = 0.9;
            this.counterfactualConfidence = motionConf * appearanceConf * herdConf * terrainConf;
        } else {
            // Standard CTRV Runge-Kutta 4 prediction
            const f_dot = (state) => {
                let [ _cx, _cy, _v, _theta, _omega ] = state;
                return [
                    _v * Math.cos(_theta),
                    _v * Math.sin(_theta),
                    0.0,
                    _omega,
                    0.0,
                    0.0,
                    0.0
                ];
            };

            let stateVec = [cx, cy, v, theta, omega, w, h];
            let k1 = f_dot(stateVec);
            let stateK2 = stateVec.map((val, idx) => val + 0.5 * dt * k1[idx]);
            let k2 = f_dot(stateK2);
            let stateK3 = stateVec.map((val, idx) => val + 0.5 * dt * k2[idx]);
            let k3 = f_dot(stateK3);
            let stateK4 = stateVec.map((val, idx) => val + dt * k3[idx]);
            let k4 = f_dot(stateK4);

            for (let i = 0; i < 7; i++) {
                stateVec[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
            }

            this.motion.cx = stateVec[0];
            this.motion.cy = stateVec[1];
            this.motion.v = stateVec[2];
            this.motion.theta = stateVec[3];
            this.motion.omega = Math.max(-0.5, Math.min(0.5, stateVec[4]));
            this.motion.w = stateVec[5];
            this.motion.h = stateVec[6];

            // Grow filter covariance P
            for (let i = 0; i < 7; i++) {
                this.motion.P[i*7 + i] += this.motion.Q_diag[i] * qScale * dt;
            }

            this.counterfactualConfidence = 1.0;
        }

        // Store memory
        this.memory.trajectoryHistory.push({ cx: this.motion.cx, cy: this.motion.cy, state: this.state });
        if (this.memory.trajectoryHistory.length > 200) {
            this.memory.trajectoryHistory.shift();
        }

        return this.stateToBbox();
    }

    update(bbox, appearanceHist, frameIdx = 0, matchCost = 0.0) {
        let z = [
            bbox[0] + (bbox[2] - bbox[0]) / 2.0,
            bbox[1] + (bbox[3] - bbox[1]) / 2.0,
            bbox[2] - bbox[0],
            bbox[3] - bbox[1]
        ];

        // Calculate physical displacement from last matched detection centroid
        let dx_phys = z[0] - this.lastUpdatedCx;
        let dy_phys = z[1] - this.lastUpdatedCy;
        let dist_phys = Math.hypot(dx_phys, dy_phys);

        // Calculate elapsed frames since last successful match
        let elapsedFrames = 1;
        if (frameIdx > 0 && this.identity.lastMatchedFrame > 0) {
            elapsedFrames = Math.max(1, frameIdx - this.identity.lastMatchedFrame);
        }

        // Correct velocity based on actual displacement and frames elapsed
        if (dist_phys > 1.0) {
            let prevTheta = this.motion.theta;
            let newTheta = Math.atan2(dy_phys, dx_phys);
            let newSpeed = dist_phys / elapsedFrames;

            // Track recent headings and speeds in behavior history
            this.behavior.headingHistory.push(newTheta);
            this.behavior.speedHistory.push(newSpeed);
            if (this.behavior.headingHistory.length > 8) this.behavior.headingHistory.shift();
            if (this.behavior.speedHistory.length > 8) this.behavior.speedHistory.shift();

            // Calculate smoothed omega using circular difference mean over the window
            let history = this.behavior.headingHistory;
            let omega = 0.0;
            if (history.length >= 2) {
                let diffs = [];
                for (let idx = 0; idx < history.length - 1; idx++) {
                    let diff = history[idx + 1] - history[idx];
                    diff = ((diff + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
                    diffs.push(diff);
                }
                omega = diffs.reduce((sum, d) => sum + d, 0) / diffs.length;
            }

            this.motion.theta = newTheta;
            this.motion.v = newSpeed;
            this.motion.omega = omega;
        } else {
            this.motion.v *= 0.8; // Decelerate if stationary
            this.motion.omega *= 0.8; // Decelerate turn rate if stationary

            this.behavior.speedHistory.push(this.motion.v);
            if (this.behavior.speedHistory.length > 8) this.behavior.speedHistory.shift();
        }

        // EKF Update with dynamic Kalman gain
        let K_gain = Math.max(0.6, Math.min(1.0, this.motion.P[0] / (this.motion.P[0] + this.motion.R_diag[0])));
        
        let dx = z[0] - this.motion.cx;
        let dy = z[1] - this.motion.cy;

        // Apply EKF state updates
        this.motion.cx = this.motion.cx + K_gain * dx;
        this.motion.cy = this.motion.cy + K_gain * dy;
        this.motion.w = this.motion.w + K_gain * (z[2] - this.motion.w);
        this.motion.h = this.motion.h + K_gain * (z[3] - this.motion.h);

        // Update last updated center coordinates to the new post-update position
        this.lastUpdatedCx = this.motion.cx;
        this.lastUpdatedCy = this.motion.cy;

        // Reset filter covariance P diagonal to lower uncertainty
        this.motion.P[0] = 5.0;  // cx
        this.motion.P[8] = 5.0;  // cy
        this.motion.P[40] = 5.0; // w
        this.motion.P[48] = 5.0; // h

        // Reset amodal uncertainty Sigma
        this.motion.Sigma = [5.0, 0.0, 0.0, 5.0];

        // Reset occlusion counter
        this.occlusion.framesOccluded = 0;
        this.identity.lastMatchedFrame = frameIdx;

        // Transition track to VISIBLE
        let wasAmodal = (this.state === "OCCLUDED" || this.state === "REMERGING" || this.state === "SEARCH");
        this.state = "VISIBLE";

        // Identity Memory Update Rule: Freeze memory except reliability if match confidence is low
        let matchConfidence = 1.0 - matchCost;
        let freeze = matchConfidence <= 0.65;

        // Update modular identityMemory
        let curAR = (bbox[2] - bbox[0]) / (bbox[3] - bbox[1]);
        this.identityMemory.update(
            appearanceHist, 
            appearanceHist, 
            curAR, 
            this.motion.v, 
            this.motion.omega, 
            matchConfidence, 
            wasAmodal,
            freeze
        );

        // Update identity embedding profile via EMA (only if not frozen)
        if (appearanceHist && !freeze) {
            if (!this.identity.appearanceEmbedding) {
                this.identity.appearanceEmbedding = appearanceHist;
            } else {
                let sim = compareHistograms(this.identity.appearanceEmbedding, appearanceHist);
                if (sim > 0.65) {
                    for(let i = 0; i < this.identity.appearanceEmbedding.length; i++) {
                        this.identity.appearanceEmbedding[i] = 0.9 * this.identity.appearanceEmbedding[i] + 0.1 * appearanceHist[i];
                    }
                }
            }
        }
    }

    stateToBbox() {
        let cx = this.motion.cx;
        let cy = this.motion.cy;
        let w = this.motion.w;
        let h = this.motion.h;
        return [cx - w/2, cy - h/2, cx + w/2, cy + h/2];
    }
}

// Helpers
function compareHistograms(h1, h2) {
    if (!h1 || !h2) return 0.5;
    let sumMin = 0;
    let sumH1 = 0;
    for (let i = 0; i < h1.length; i++) {
        sumMin += Math.min(h1[i], h2[i]);
        sumH1 += h1[i];
    }
    return sumH1 > 0 ? (sumMin / sumH1) : 0.0;
}

function calculateIoU(box1, box2) {
    let xi1 = Math.max(box1[0], box2[0]);
    let yi1 = Math.max(box1[1], box2[1]);
    let xi2 = Math.min(box1[2], box2[2]);
    let yi2 = Math.min(box1[3], box2[3]);
    let interArea = Math.max(0.0, xi2 - xi1) * Math.max(0.0, yi2 - yi1);
    let box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1]);
    let box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1]);
    let unionArea = box1Area + box2Area - interArea;
    return unionArea > 0 ? (interArea / unionArea) : 0.0;
}

// --- 3.5 Kuhn-Munkres (Hungarian) Assignment Algorithm ---
function minWeightBipartiteMatching(costMatrix) {
    let numRows = costMatrix.length;
    let numCols = costMatrix[0] ? costMatrix[0].length : 0;
    if (numRows === 0 || numCols === 0) return [];

    let transposed = false;
    let matrix = costMatrix;
    if (numRows > numCols) {
        transposed = true;
        matrix = Array.from({ length: numCols }, (_, c) => 
            Array.from({ length: numRows }, (_, r) => costMatrix[r][c])
        );
        let temp = numRows;
        numRows = numCols;
        numCols = temp;
    }

    let u = new Array(numRows + 1).fill(0);
    let v = new Array(numCols + 1).fill(0);
    let p = new Array(numCols + 1).fill(0);
    let way = new Array(numCols + 1).fill(0);

    for (let i = 1; i <= numRows; i++) {
        p[0] = i;
        let j0 = 0;
        let minv = new Array(numCols + 1).fill(Infinity);
        let used = new Array(numCols + 1).fill(false);
        do {
            used[j0] = true;
            let i0 = p[j0];
            let delta = Infinity;
            let j1 = 0;
            for (let j = 1; j <= numCols; j++) {
                if (!used[j]) {
                    let cur = matrix[i0 - 1][j - 1] - u[i0] - v[j];
                    if (cur < minv[j]) {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if (minv[j] < delta) {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for (let j = 0; j <= numCols; j++) {
                if (used[j]) {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
        } while (p[j0] !== 0);

        do {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
        } while (j0 !== 0);
    }

    let matches = [];
    for (let j = 1; j <= numCols; j++) {
        if (p[j] > 0) {
            let r = p[j] - 1;
            let c = j - 1;
            if (transposed) {
                matches.push([c, r]);
            } else {
                matches.push([r, c]);
            }
        }
    }
    return matches;
}

// --- 3.6 Modular Identity Memory Structure ---
class AppearanceMemory {
    constructor(embedding, colorHist, aspectRatio) {
        this.averageEmbedding = embedding ? [...embedding] : null;
        this.averageColorHist = colorHist ? [...colorHist] : null;
        this.averageBodyRatio = aspectRatio;
        this.textureEMA = null;
        this.semanticBodyAttributes = null; // e.g. horn morphology, coat pattern, facial markings
    }
    update(embedding, colorHist, aspectRatio) {
        if (embedding) {
            if (!this.averageEmbedding) {
                this.averageEmbedding = [...embedding];
            } else {
                for (let i = 0; i < this.averageEmbedding.length; i++) {
                    this.averageEmbedding[i] = 0.95 * this.averageEmbedding[i] + 0.05 * embedding[i];
                }
            }
        }
        if (colorHist) {
            if (!this.averageColorHist) {
                this.averageColorHist = [...colorHist];
            } else {
                for (let i = 0; i < this.averageColorHist.length; i++) {
                    this.averageColorHist[i] = 0.95 * this.averageColorHist[i] + 0.05 * colorHist[i];
                }
            }
        }
        if (aspectRatio) {
            this.averageBodyRatio = 0.95 * this.averageBodyRatio + 0.05 * aspectRatio;
        }
    }
}

class MotionMemory {
    constructor(speed, turnRate) {
        this.averageSpeed = speed;
        this.averageAcceleration = 0.0;
        this.averageTurnRate = turnRate;
        this.prevSpeed = speed;
    }
    update(speed, turnRate) {
        let acc = speed - this.prevSpeed;
        this.averageSpeed = 0.95 * this.averageSpeed + 0.05 * speed;
        this.averageAcceleration = 0.95 * this.averageAcceleration + 0.05 * acc;
        this.averageTurnRate = 0.95 * this.averageTurnRate + 0.05 * turnRate;
        this.prevSpeed = speed;
    }
}

class BehaviouralDynamicsMemory {
    constructor() {
        this.averageGrazingSpeed = 1.0;
        this.stopFrequency = 0.05;
        this.turnFrequency = 0.05;
        this.restingRatio = 0.1;
        this.movementEntropy = 0.5;
        this.speedHistory = [];
        this.restingDuration = 0;
    }
    update(speed, turnRate) {
        if (speed < 0.2) {
            this.restingDuration++;
        } else {
            this.restingDuration = 0;
        }
        
        let isTurning = Math.abs(turnRate) > 0.05 ? 1.0 : 0.0;
        this.turnFrequency = 0.98 * this.turnFrequency + 0.02 * isTurning;
        
        if (speed >= 0.2) {
            this.averageGrazingSpeed = 0.95 * this.averageGrazingSpeed + 0.05 * speed;
        }
        
        let isStopped = speed < 0.2 ? 1.0 : 0.0;
        this.stopFrequency = 0.98 * this.stopFrequency + 0.02 * isStopped;
        
        this.speedHistory.push(speed);
        if (this.speedHistory.length > 50) {
            this.speedHistory.shift();
        }
        
        let stoppedCount = this.speedHistory.filter(s => s < 0.2).length;
        this.restingRatio = this.speedHistory.length > 0 ? (stoppedCount / this.speedHistory.length) : 0.0;
        
        // movementEntropy calculation (mean and std dev)
        if (this.speedHistory.length > 0) {
            let mean = this.speedHistory.reduce((a, b) => a + b, 0) / this.speedHistory.length;
            let variance = this.speedHistory.reduce((a, b) => a + Math.pow(b - mean, 2), 0) / this.speedHistory.length;
            let stdDev = Math.sqrt(variance);
            this.movementEntropy = stdDev / (mean + 1e-6);
        } else {
            this.movementEntropy = 0.0;
        }
    }
}

class SocialMemory {
    constructor() {
        this.nearestNeighbours = [];
        this.leader = null;
        this.follower = null;
        this.herdAffiliation = 1.0;
    }
    update(neighbours, leader, follower) {
        this.nearestNeighbours = neighbours || [];
        this.leader = leader || null;
        this.follower = follower || null;
    }
}

class ReliabilityMemory {
    constructor(initialConf) {
        this.successfulMatches = 1;
        this.successfulRecoveries = 0;
        this.falseRecoveries = 0;
        this.reliabilityHistory = [initialConf];
        this.averageReliability = initialConf;
    }
    update(reliability, recovered = false) {
        this.successfulMatches++;
        if (recovered) {
            this.successfulRecoveries++;
        }
        this.reliabilityHistory.push(reliability);
        if (this.reliabilityHistory.length > 50) {
            this.reliabilityHistory.shift();
        }
        this.averageReliability = this.reliabilityHistory.reduce((sum, c) => sum + c, 0) / this.reliabilityHistory.length;
    }
}

class IdentityMemory {
    constructor(embedding, colorHist, aspectRatio, speed, turnRate, initialConf) {
        this.appearance = new AppearanceMemory(embedding, colorHist, aspectRatio);
        this.motion = new MotionMemory(speed, turnRate);
        this.behaviour = new BehaviouralDynamicsMemory();
        this.social = new SocialMemory();
        this.reliability = new ReliabilityMemory(initialConf);
    }
    update(embedding, colorHist, aspectRatio, speed, turnRate, reliability, recovered = false, freeze = false) {
        this.reliability.update(reliability, recovered);
        if (!freeze) {
            this.appearance.update(embedding, colorHist, aspectRatio);
            this.motion.update(speed, turnRate);
            this.behaviour.update(speed, turnRate);
        }
    }
}

// --- 4. Counterfactual Amodal Tracker (CAT) ---

class CounterfactualAmodalTracker {
    constructor(projector) {
        this.projector = projector;
        this.tracks = [];
        this.nextId = 1;
        this.frameCount = 0;
        this.prevCamOffset = { x: 0, y: 0 };
    }

    // Camera Motion Compensation
    applyCameraMotionCompensation(camOffset) {
        let dx = camOffset.x - this.prevCamOffset.x;
        let dy = camOffset.y - this.prevCamOffset.y;
        
        for (let track of this.tracks) {
            track.motion.cx -= dx;
            track.motion.cy -= dy;
        }
        this.prevCamOffset = { ...camOffset };
    }

    // Expose getters for legacy evaluateMOT() and draw() drawing blocks
    get amodalAnchors() {
        let anchors = {};
        for (let track of this.tracks) {
            if (track.state === "OCCLUDED" || track.state === "REMERGING") {
                anchors[track.trackId] = {
                    cx: track.motion.cx,
                    cy: track.motion.cy,
                    w: track.motion.w,
                    h: track.motion.h,
                    Sigma: track.motion.Sigma,
                    framesOccluded: track.occlusion.framesOccluded,
                    velocity: [track.motion.v * Math.cos(track.motion.theta), track.motion.v * Math.sin(track.motion.theta)],
                    isExpired: track.state === "EXPIRED",
                    appearanceEmbedding: track.identity.appearanceEmbedding
                };
            }
        }
        return anchors;
    }

    get baselineTracker() {
        return {
            tracks: this.tracks.map(t => {
                return {
                    trackId: t.trackId,
                    x: [t.motion.cx, t.motion.cy, t.motion.v, t.motion.theta, t.motion.omega, t.motion.w, t.motion.h],
                    timeSinceUpdate: (t.state === "VISIBLE" || t.state === "NEW") ? 0 : 1,
                    lastBbox: t.stateToBbox(),
                    appearanceEmbedding: t.identity.appearanceEmbedding
                };
            })
        };
    }

    step(detections, detHistograms, vegetationMaskCanvas, uavAlt, uavPitch, maxTimeout = 150, cohesionWeight = 0.5, camOffset = null) {
        this.frameCount++;
        triggerLog("sys", `---------------------------------------------`);
        triggerLog("sys", `[DIAGNOSTIC] === FRAME ${this.frameCount} ===`);
        for (let track of this.tracks) {
            let tsu = (track.state === "VISIBLE" || track.state === "NEW") ? 0 : (this.frameCount - track.identity.lastMatchedFrame);
            triggerLog("sys", `[DIAGNOSTIC] Pre-Step: ID ${track.trackId} | state=${track.state} | tsu=${tsu} | pos=(${track.motion.cx.toFixed(1)}, ${track.motion.cy.toFixed(1)})`);
        }

        if (camOffset) {
            this.applyCameraMotionCompensation(camOffset);
        }

        // 1. Calculate visible herd mean velocity for social herd prior
        let visibleVelocities = [];
        for (let track of this.tracks) {
            if (track.state === "VISIBLE" || track.state === "NEW") {
                let vx = track.motion.v * Math.cos(track.motion.theta);
                let vy = track.motion.v * Math.sin(track.motion.theta);
                visibleVelocities.push([vx, vy]);
            }
        }
        let meanHerdVel = null;
        if (visibleVelocities.length > 0) {
            meanHerdVel = [0.0, 0.0];
            meanHerdVel[0] = visibleVelocities.reduce((sum, v) => sum + v[0], 0) / visibleVelocities.length;
            meanHerdVel[1] = visibleVelocities.reduce((sum, v) => sum + v[1], 0) / visibleVelocities.length;
        }

        // 2. Motion Prediction
        let predictedBboxes = this.tracks.map(t => t.predict(1.0, meanHerdVel, cohesionWeight));

        // 3. Build Cost Matrix for bipartite assignment (Hungarian matching)
        let numTracks = this.tracks.length;
        let numDets = detections.length;

        let costMatrix = Array.from({ length: numTracks }, () => new Float32Array(numDets));
        let candidateDiagnostics = [];

        // Define main cost coefficients based on track state
        let trackWeights = this.tracks.map(track => {
            let alpha, beta, gamma, delta, lambda;
            if (track.state === "OCCLUDED" || track.state === "REMERGING" || track.state === "SEARCH") {
                alpha = 0.15;  // motion (IoU)
                beta = 0.25;   // appearance
                gamma = 0.15;  // social
                delta = 0.20;  // counterfactual EKF
                lambda = 0.25;  // identity prior
            } else {
                alpha = 0.45;  // motion
                beta = 0.25;   // appearance
                gamma = 0.05;  // social
                delta = 0.05;  // counterfactual
                lambda = 0.20;  // identity prior
            }
            return { alpha, beta, gamma, delta, lambda };
        });

        // Compute hierarchical cost components
        for (let i = 0; i < numTracks; i++) {
            let track = this.tracks[i];
            let weights = trackWeights[i];
            let predBox = predictedBboxes[i];

            for (let j = 0; j < numDets; j++) {
                let detBox = detections[j];
                let detHist = detHistograms[j];

                // --- A. MOTION COST (Cm) ---
                let iou = calculateIoU(predBox, detBox);
                let c_motion = 1.0 - iou;

                // --- B. SEMANTIC APPEARANCE COST (Capp) ---
                let c_embed = 1.0 - compareHistograms(track.identityMemory.appearance.averageEmbedding || track.identity.appearanceEmbedding, detHist);
                let c_color = 1.0 - compareHistograms(track.identityMemory.appearance.averageColorHist || track.identity.appearanceEmbedding, detHist);
                let c_texture = 0.1; // Baseline local texture descriptor difference
                let detW = detBox[2] - detBox[0];
                let detH = detBox[3] - detBox[1];
                let detAR = detW / detH;
                let c_shape = Math.min(1.0, Math.abs(track.identityMemory.appearance.averageBodyRatio - detAR) / 0.5);
                let c_semantic = 0.1; // Baseline semantic components difference

                let c_app = 0.40 * c_embed + 0.20 * c_color + 0.15 * c_texture + 0.15 * c_shape + 0.10 * c_semantic;

                // --- C. SOCIAL PRIOR COST (Cs) ---
                let detCx = detBox[0] + (detBox[2] - detBox[0]) / 2.0;
                let detCy = detBox[1] + (detBox[3] - detBox[1]) / 2.0;
                let c_social = 0.5;
                if (meanHerdVel) {
                    let expCx = track.motion.cx + meanHerdVel[0];
                    let expCy = track.motion.cy + meanHerdVel[1];
                    let distHerd = Math.hypot(detCx - expCx, detCy - expCy);
                    c_social = Math.min(1.0, distHerd / 100.0);
                }

                // --- D. COUNTERFACTUAL ESTIMATION COST (Cc) ---
                let dx = detCx - track.motion.cx;
                let dy = detCy - track.motion.cy;
                let dist = Math.hypot(dx, dy);
                let stdX = Math.sqrt(track.motion.Sigma[0]);
                let stdY = Math.sqrt(track.motion.Sigma[3]);
                let maxDev = 3.5 * Math.max(stdX, stdY);
                let c_counterfactual = Math.min(1.0, dist / maxDev);

                // --- E. MEMORY COST (Cmemory) ---
                let c_mem_app = c_embed;
                let c_mem_mot = Math.min(1.0, Math.abs(track.motion.v - track.identityMemory.motion.averageSpeed) / Math.max(1.0, track.identityMemory.motion.averageSpeed));
                let c_mem_beh = Math.min(1.0, Math.abs(track.motion.omega - track.identityMemory.behaviour.turnFrequency));
                let c_mem_soc = 1.0 - track.identityMemory.social.herdAffiliation;
                let c_memory = 0.40 * c_mem_app + 0.30 * c_mem_mot + 0.15 * c_mem_beh + 0.15 * c_mem_soc;

                // --- F. IDENTITY PRIOR COST (Cprior) ---
                let c_age = Math.exp(-track.identity.age / 100.0);
                let c_continuity = (track.state === "OCCLUDED" || track.state === "SEARCH" || track.state === "REMERGING") ? 0.0 : 1.0;
                let c_reliability = 1.0 - track.identityMemory.reliability.averageReliability;

                // Trajectory consistency
                let c_traj = 0.5;
                let elapsedFrames = Math.max(1, this.frameCount - track.identity.lastMatchedFrame);
                let candVx = (detCx - track.lastUpdatedCx) / elapsedFrames;
                let candVy = (detCy - track.lastUpdatedCy) / elapsedFrames;
                let candSpeed = Math.hypot(candVx, candVy);
                let candTheta = Math.atan2(candVy, candVx);
                
                let headingDiff = Math.abs(candTheta - track.motion.theta);
                headingDiff = ((headingDiff + Math.PI) % (2 * Math.PI) + 2 * Math.PI) % (2 * Math.PI) - Math.PI;
                let c_traj_heading = 0.5 - 0.5 * Math.cos(headingDiff);
                let avgSpeed = track.identityMemory.motion.averageSpeed;
                let c_traj_speed = Math.min(1.0, Math.abs(candSpeed - avgSpeed) / Math.max(1.0, avgSpeed));
                c_traj = 0.5 * c_traj_heading + 0.5 * c_traj_speed;

                let c_prior = 0.20 * c_age + 0.20 * c_continuity + 0.15 * c_reliability + 0.20 * c_traj + 0.25 * c_memory;

                // --- FINAL GLOBAL COST (C) ---
                let totalCost = weights.alpha * c_motion + weights.beta * c_app + weights.gamma * c_social + weights.delta * c_counterfactual + weights.lambda * c_prior;
                costMatrix[i][j] = totalCost;

                candidateDiagnostics.push({
                    trackId: track.trackId,
                    trackState: track.state,
                    detIdx: j,
                    motion: c_motion,
                    app: { total: c_app, embed: c_embed, color: c_color, texture: c_texture, shape: c_shape, semantic: c_semantic },
                    social: c_social,
                    counterfactual: c_counterfactual,
                    prior: { total: c_prior, age: c_age, continuity: c_continuity, reliability: c_reliability, traj: c_traj, memory: c_memory },
                    totalCost: totalCost,
                    penalized: false
                });
            }
        }

        // Apply Counterfactual Duplicate Correction
        for (let i = 0; i < numTracks; i++) {
            let trackA = this.tracks[i];
            if (trackA.state === "OCCLUDED" || trackA.state === "SEARCH" || trackA.state === "REMERGING") {
                for (let k = 0; k < numTracks; k++) {
                    let trackB = this.tracks[k];
                    if (trackB.trackId > trackA.trackId && (trackB.state === "VISIBLE" || trackB.state === "NEW")) {
                        let iou = calculateIoU(predictedBboxes[i], predictedBboxes[k]);
                        let dist = Math.hypot(trackA.motion.cx - trackB.motion.cx, trackA.motion.cy - trackB.motion.cy);
                        if (iou > 0.2 || dist < 80.0) {
                            for (let j = 0; j < numDets; j++) {
                                let detHist = detHistograms[j];
                                let simA = compareHistograms(trackA.identityMemory.appearance.averageEmbedding || trackA.identity.appearanceEmbedding, detHist);
                                if (simA > 0.65) {
                                    // Apply counterfactual prioritization penalty to continuation track
                                    costMatrix[k][j] += 0.8;
                                    let diag = candidateDiagnostics.find(d => d.trackId === trackB.trackId && d.detIdx === j);
                                    if (diag) {
                                        diag.totalCost += 0.8;
                                        diag.penalized = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        // Solve association globally via Hungarian Matcher
        let matches = minWeightBipartiteMatching(costMatrix);
        let matchedTracks = new Set();
        let matchedDets = new Set();

        for (let [r, c] of matches) {
            let track = this.tracks[r];
            let detBox = detections[c];
            let detHist = detHistograms[c];

            let iou = calculateIoU(predictedBboxes[r], detBox);
            let sim = compareHistograms(track.identityMemory.appearance.averageEmbedding || track.identity.appearanceEmbedding, detHist);
            
            let detCx = detBox[0] + (detBox[2] - detBox[0]) / 2.0;
            let detCy = detBox[1] + (detBox[3] - detBox[1]) / 2.0;
            let dx = detCx - track.motion.cx;
            let dy = detCy - track.motion.cy;
            let dist = Math.hypot(dx, dy);

            let stdX = Math.sqrt(track.motion.Sigma[0]);
            let stdY = Math.sqrt(track.motion.Sigma[3]);
            let maxDev = 3.5 * Math.max(stdX, stdY);

            let isMatched = false;
            if (track.state === "OCCLUDED" || track.state === "REMERGING" || track.state === "SEARCH") {
                if (iou > 0.15 || (sim > 0.65 && dist <= maxDev)) {
                    isMatched = true;
                    track.state = "REMERGING";
                    triggerLog("remerge", `Track ${track.trackId} re-emerged with similarity score ${sim.toFixed(2)} (IoU: ${iou.toFixed(2)}, dist: ${dist.toFixed(1)}px)`);
                    updateReIDPanel(track.trackId, track.identity.appearanceEmbedding, detHist, sim);
                }
            } else {
                if (iou > 0.3 || (sim > 0.65 && (iou > 0.05 || dist < 60.0))) {
                    isMatched = true;
                }
            }

            if (isMatched) {
                let matchCost = costMatrix[r][c];
                track.update(detBox, detHist, this.frameCount, matchCost);
                matchedTracks.add(r);
                matchedDets.add(c);
                triggerLog("sys", `[CAA] Matched detection index ${c} -> Track ${track.trackId}`);
                triggerLog("sys", `[CAA] Track ${track.trackId} state set: ACTIVE, timeSinceUpdate=0, pos=(${track.motion.cx.toFixed(1)}, ${track.motion.cy.toFixed(1)})`);
            }
        }

        // Output candidate score diagnostics
        for (let j = 0; j < numDets; j++) {
            triggerLog("sys", `Candidate scores for Detection ${j}:`);
            let detDiags = candidateDiagnostics.filter(d => d.detIdx === j);
            for (let d of detDiags) {
                let penStr = d.penalized ? " [PENALIZED]" : "";
                triggerLog("sys", `  Track ${d.trackId} (${d.trackState}): Cm=${d.motion.toFixed(3)}, Capp=${d.app.total.toFixed(3)} (embed=${d.app.embed.toFixed(2)}, color=${d.app.color.toFixed(2)}, shape=${d.app.shape.toFixed(2)}), Cs=${d.social.toFixed(3)}, Cc=${d.counterfactual.toFixed(3)}, Cprior=${d.prior.total.toFixed(3)} (age=${d.prior.age.toFixed(2)}, continuity=${d.prior.continuity.toFixed(2)}, reliability=${d.prior.reliability.toFixed(2)}, trajectory=${d.prior.traj.toFixed(2)}, memory=${d.prior.memory.toFixed(2)}), TOTAL=${d.totalCost.toFixed(3)}${penStr}`);
            }
        }

        for (let i = 0; i < this.tracks.length; i++) {
            let track = this.tracks[i];
            let matched = matchedTracks.has(i);
            let tsu = (track.state === "VISIBLE" || track.state === "NEW") ? 0 : (this.frameCount - track.identity.lastMatchedFrame);
            triggerLog("sys", `[DIAGNOSTIC] Post-Association: ID ${track.trackId} | state=${track.state} | matched=${matched} | tsu=${tsu}`);
        }

        // 4. Handle unmatched tracks (State Transitions)
        let ctx = vegetationMaskCanvas.getContext('2d');
        for (let i = 0; i < this.tracks.length; i++) {
            if (matchedTracks.has(i)) continue;

            let track = this.tracks[i];
            
            if (track.state === "VISIBLE" || track.state === "NEW") {
                let [x1, y1, x2, y2] = predictedBboxes[i];
                let cx = Math.floor(x1 + (x2 - x1) / 2);
                let cy = Math.floor(y1 + (y2 - y1) / 2);

                let maskX = cx;
                let maskY = cy;
                if (camOffset) {
                    maskX = Math.floor(cx + camOffset.x);
                    maskY = Math.floor(cy + camOffset.y);
                }

                let isOccluded = false;
                let rx = Math.max(2, Math.floor(track.motion.w * 0.15));
                let ry = Math.max(2, Math.floor(track.motion.h * 0.15));
                
                let points = [
                    [maskX, maskY],
                    [maskX - rx, maskY],
                    [maskX + rx, maskY],
                    [maskX, maskY - ry],
                    [maskX, maskY + ry]
                ];
                
                for (let pt of points) {
                    let px = pt[0];
                    let py = pt[1];
                    if (px >= 0 && px < vegetationMaskCanvas.width && py >= 0 && py < vegetationMaskCanvas.height) {
                        let pixel = ctx.getImageData(px, py, 1, 1).data;
                        if (pixel[1] > 50 && pixel[3] > 0) {
                            isOccluded = true;
                            break;
                        }
                    }
                }

                if (isOccluded) {
                    track.state = "OCCLUDED";
                    track.occlusion.entryFrame = this.frameCount;
                    track.identityMemory.reliability.successfulRecoveries = 0; // reset
                    triggerLog("seed", `Track ${track.trackId} entered occlusion. Seeding virtual anchor.`);
                } else {
                    track.state = "LOST";
                    triggerLog("sys", `[CAT] Track ${track.trackId} lost observation. Transition: LOST`);
                }
            } else if (track.state === "OCCLUDED") {
                if (track.occlusion.framesOccluded > 15) {
                    track.state = "SEARCH";
                    triggerLog("sys", `[CAT] Track ${track.trackId} occluded >15 frames. Transition: SEARCH`);
                }
            } else if (track.state === "LOST") {
                let timeLost = this.frameCount - track.identity.lastMatchedFrame;
                if (timeLost > 15) {
                    track.state = "SEARCH";
                    triggerLog("sys", `[CAT] Track ${track.trackId} missing >15 frames. Transition: SEARCH`);
                }
            } else if (track.state === "SEARCH") {
                let timeLost = this.frameCount - track.identity.lastMatchedFrame;
                let velMag = Math.hypot(track.motion.v * Math.cos(track.motion.theta), track.motion.v * Math.sin(track.motion.theta));
                let timeoutLimit = velMag < 0.3 ? (maxTimeout * 2) : maxTimeout;
                if (timeLost > timeoutLimit) {
                    track.state = "EXPIRED";
                    triggerLog("sys", `[CAT] Track ${track.trackId} expired outside occlusion. Transition: EXPIRED`);
                }
            }
        }

        // 5. Spawn new tracks for unmatched detections
        let unmatchedDetections = [];
        let matchedDetections = [];
        for (let j = 0; j < detections.length; j++) {
            if (matchedDets.has(j)) {
                matchedDetections.push(j);
            } else {
                unmatchedDetections.push(j);
            }
        }
        triggerLog("sys", `[DIAGNOSTIC] Detections count: ${detections.length}. Matched detections: [${matchedDetections.join(', ')}], Unmatched detections: [${unmatchedDetections.join(', ')}]`);

        for (let j = 0; j < detections.length; j++) {
            if (!matchedDets.has(j)) {
                let newTrack = new CounterfactualAmodalTrack(detections[j], this.nextId++, detHistograms[j]);
                newTrack.state = "VISIBLE";
                newTrack.identity.lastMatchedFrame = this.frameCount;
                this.tracks.push(newTrack);
                triggerLog("sys", `[CAT] Spawned new Track ID ${newTrack.trackId} from unmatched detection.`);
            }
        }

        // 6. Duplicate anchor suppression and merging
        let toSuppress = new Set();
        for (let i = 0; i < this.tracks.length; i++) {
            for (let j = i + 1; j < this.tracks.length; j++) {
                let t1 = this.tracks[i];
                let t2 = this.tracks[j];
                
                let box1 = t1.stateToBbox();
                let box2 = t2.stateToBbox();
                let iou = calculateIoU(box1, box2);
                
                if (iou > 0.6) {
                    let sim = compareHistograms(t1.identity.appearanceEmbedding, t2.identity.appearanceEmbedding);
                    if (sim > 0.65) {
                        let older = t1.trackId < t2.trackId ? t1 : t2;
                        let newer = t1.trackId < t2.trackId ? t2 : t1;
                        
                        // If the newer track is visible and the older track is occluded/lost,
                        // the older track inherits the state and position to prevent identity switches.
                        if ((newer.state === "VISIBLE" || newer.state === "NEW") && 
                            older.state !== "VISIBLE" && older.state !== "NEW") {
                            older.state = newer.state;
                            older.motion.cx = newer.motion.cx;
                            older.motion.cy = newer.motion.cy;
                            older.motion.v = newer.motion.v;
                            older.motion.theta = newer.motion.theta;
                            older.motion.omega = newer.motion.omega;
                            older.motion.w = newer.motion.w;
                            older.motion.h = newer.motion.h;
                            older.motion.P = [...newer.motion.P];
                            older.motion.Sigma = [...newer.motion.Sigma];
                            older.identity.appearanceEmbedding = newer.identity.appearanceEmbedding;
                            older.identity.lastMatchedFrame = newer.identity.lastMatchedFrame;
                            older.lastUpdatedCx = newer.lastUpdatedCx;
                            older.lastUpdatedCy = newer.lastUpdatedCy;
                        }
                        
                        toSuppress.add(newer.trackId);
                        triggerLog("sys", `Merged duplicate track ${newer.trackId} into older track ${older.trackId}.`);
                    }
                }
            }
        }
        this.tracks = this.tracks.filter(t => !toSuppress.has(t.trackId));

        // 7. Cleanup EXPIRED tracks
        this.tracks = this.tracks.filter(t => t.state !== "EXPIRED");

        for (let track of this.tracks) {
            let tsu = (track.state === "VISIBLE" || track.state === "NEW") ? 0 : (this.frameCount - track.identity.lastMatchedFrame);
            triggerLog("sys", `[DIAGNOSTIC] Post-Cleanup: ID ${track.trackId} | state=${track.state} | tsu=${tsu}`);
        }
        triggerLog("sys", `[DIAGNOSTIC] === END FRAME ${this.frameCount} ===`);

        // 8. Compile outputs list
        let output = [];
        for (let track of this.tracks) {
            let bbox = track.stateToBbox();
            if (track.state === "VISIBLE" || track.state === "NEW") {
                output.push({
                    trackId: track.trackId,
                    bbox: bbox,
                    status: "visible",
                    velocity: [track.motion.v * Math.cos(track.motion.theta), track.motion.v * Math.sin(track.motion.theta)]
                });
            } else if (track.state === "OCCLUDED" || track.state === "REMERGING") {
                output.push({
                    trackId: track.trackId,
                    bbox: bbox,
                    status: "occluded_virtual",
                    sigma: track.motion.Sigma,
                    framesOccluded: track.occlusion.framesOccluded,
                    velocity: [track.motion.v * Math.cos(track.motion.theta), track.motion.v * Math.sin(track.motion.theta)]
                });
            }
        }
        
        // Log status for debugging
        triggerLog("sys", `[BASELINE UPDATE] Detections entering baseline: ${detections.length}. activeAnchorIds: [${Array.from(new Set(this.tracks.filter(t => t.state === "OCCLUDED" || t.state === "REMERGING").map(t => t.trackId))).join(', ')}]`);
        triggerLog("sys", `[BASELINE UPDATE] Active visible tracks returned: ${output.filter(t => t.status === "visible").map(t => t.trackId).join(', ')}`);
        for (let track of this.tracks) {
            let timeSinceUpdate = (track.state === "VISIBLE" || track.state === "NEW") ? 0 : (this.frameCount - track.identity.lastMatchedFrame);
            triggerLog("sys", `[TRACK STATUS] ID ${track.trackId}: state=${track.state}, timeSinceUpdate=${timeSinceUpdate}`);
        }

        return output;
    }
}

// --- 7. Simulation Scenarios & Cow Generator ---
const Scenarios = {
    linear: {
        title: "Linear Crossing",
        description: "A single cow walks in a straight path, gets occluded by a tree canopy, and re-emerges.",
        init(simulator) {
            simulator.cows = [
                {
                    id: 1,
                    cx: 100, cy: 300,
                    vx: 2.2, vy: 0.1,
                    w: 52, h: 42,
                    color: "holstein",
                    // Simulating a strong HSV-based appearance histogram (r, g, b components mock)
                    hist: [0.1, 0.15, 0.45, 0.2, 0.05, 0.02, 0.01, 0.02],
                    path: []
                }
            ];
            simulator.trees = [
                { cx: 500, cy: 300, r: 90 }
            ];
            simulator.maxFrames = 400;
        }
    },
    curved: {
        title: "Curved CTRV path",
        description: "A cow performs a sweeping turn inside the thicket. EKF turn rate (omega) predicts the curve.",
        init(simulator) {
            simulator.cows = [
                {
                    id: 1,
                    cx: 120, cy: 150,
                    v: 2.4, theta: 0.2, omega: 0.009, // starts moving down-right and curving
                    w: 52, h: 42,
                    color: "brown",
                    hist: [0.35, 0.25, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05],
                    path: []
                }
            ];
            simulator.trees = [
                { cx: 480, cy: 260, r: 120 }
            ];
            simulator.maxFrames = 400;
        }
    },
    herd: {
        title: "Herd Cohesion (Priors)",
        description: "An occluded cow turns inside the canopy to follow the visible herd, guided by blended velocity.",
        init(simulator) {
            // Cow 1 (Herd Leader, visible): circles around the bottom tree
            // Cow 2 (Occluded, enters tree): turns upward following the leader's mean velocity vector
            simulator.cows = [
                {
                    id: 1, // Visible leader
                    cx: 150, cy: 400,
                    vx: 2.2, vy: -0.1,
                    w: 50, h: 40,
                    color: "holstein",
                    hist: [0.1, 0.2, 0.4, 0.15, 0.05, 0.05, 0.02, 0.03],
                    path: [],
                    updateBehavior(frameIdx) {
                        // Leader turns upwards after frame 130
                        if (frameIdx > 130 && frameIdx < 220) {
                            this.vy = -1.2;
                            this.vx = 1.4;
                        } else if (frameIdx >= 220) {
                            this.vy = -0.1;
                            this.vx = 2.0;
                        }
                    }
                },
                {
                    id: 2, // Occluded cow
                    cx: 120, cy: 260,
                    vx: 2.2, vy: 0.0,
                    w: 50, h: 40,
                    color: "black",
                    hist: [0.05, 0.05, 0.05, 0.05, 0.2, 0.3, 0.2, 0.1],
                    path: [],
                    updateBehavior(frameIdx) {
                        // Enters tree canopy. While occluded, it follows the herd leader's turn
                        if (frameIdx > 135 && frameIdx < 230) {
                            this.vy = -0.9;
                            this.vx = 1.3;
                        } else if (frameIdx >= 230) {
                            this.vy = 0.0;
                            this.vx = 2.0;
                        }
                    }
                }
            ];
            simulator.trees = [
                { cx: 480, cy: 240, r: 90 }
            ];
            simulator.maxFrames = 400;
        }
    },
    resting: {
        title: "Stationary Grazing",
        description: "A resting cow stops inside the canopy. Dynamic timeout extends tracking to 300 frames.",
        init(simulator) {
            simulator.cows = [
                {
                    id: 1,
                    cx: 150, cy: 280,
                    vx: 2.0, vy: 0.0,
                    w: 54, h: 44,
                    color: "holstein",
                    hist: [0.15, 0.15, 0.4, 0.1, 0.05, 0.05, 0.05, 0.05],
                    path: [],
                    updateBehavior(frameIdx) {
                        // Decelerate and stop inside the tree (frame 140 to 300)
                        if (frameIdx >= 140 && frameIdx < 160) {
                            this.vx *= 0.7;
                        } else if (frameIdx >= 160 && frameIdx < 280) {
                            this.vx = 0.005; // almost stationary
                        } else if (frameIdx >= 280) {
                            this.vx = 2.0; // wakes up and emerges
                        }
                    }
                }
            ];
            simulator.trees = [
                { cx: 450, cy: 280, r: 100 }
            ];
            simulator.maxFrames = 450;
        }
    },
    complete_herd: {
        title: "Complete Herd Occlusion",
        description: "A group of 5 cows walk together and get fully occluded simultaneously. CAT preserves all 5 identities.",
        init(simulator) {
            simulator.cows = [
                { id: 1, cx: 100, cy: 220, vx: 2.1, vy: 0.1, w: 50, h: 40, color: "holstein", hist: [0.1, 0.2, 0.4, 0.1, 0.05, 0.05, 0.05, 0.05], path: [] },
                { id: 2, cx: 80,  cy: 280, vx: 2.1, vy: 0.1, w: 52, h: 42, color: "brown",    hist: [0.35, 0.25, 0.15, 0.05, 0.05, 0.05, 0.05, 0.05], path: [] },
                { id: 3, cx: 120, cy: 340, vx: 2.1, vy: 0.1, w: 48, h: 38, color: "black",    hist: [0.05, 0.05, 0.05, 0.05, 0.2, 0.3, 0.2, 0.1], path: [] },
                { id: 4, cx: 60,  cy: 310, vx: 2.1, vy: 0.1, w: 50, h: 40, color: "holstein", hist: [0.12, 0.18, 0.38, 0.12, 0.05, 0.05, 0.05, 0.05], path: [] },
                { id: 5, cx: 140, cy: 250, vx: 2.1, vy: 0.1, w: 52, h: 42, color: "brown",    hist: [0.3, 0.3, 0.2, 0.05, 0.05, 0.05, 0.02, 0.03], path: [] }
            ];
            simulator.trees = [
                { cx: 500, cy: 280, r: 140 }
            ];
            simulator.maxFrames = 450;
        }
    },
    sandbox: {
        title: "Interactive Sandbox",
        description: "Double-click to add a tree. Drag the cow or draw a path with your mouse.",
        init(simulator) {
            simulator.cows = [
                {
                    id: 1,
                    cx: 150, cy: 200,
                    vx: 1.5, vy: 0.8,
                    w: 50, h: 40,
                    color: "holstein",
                    hist: [0.1, 0.2, 0.4, 0.15, 0.05, 0.05, 0.02, 0.03],
                    path: []
                }
            ];
            simulator.trees = [
                { cx: 400, cy: 300, r: 85 }
            ];
            simulator.maxFrames = Infinity;
        }
    }
};

// --- 8. Core Simulator Control Engine ---
class OcclusionSimulator {
    constructor() {
        this.uavCanvas = document.getElementById("uav-canvas");
        this.projCanvas = document.getElementById("projection-canvas");
        
        this.uavCtx = this.uavCanvas.getContext("2d");
        this.projCtx = this.projCanvas.getContext("2d");

        // Invisible canvas to compute ExG vegetation segmentation binary mask in pixel space
        this.maskCanvas = document.createElement("canvas");
        this.maskCanvas.width = this.uavCanvas.width;
        this.maskCanvas.height = this.uavCanvas.height;
        this.maskCtx = this.maskCanvas.getContext("2d");

        this.projector = new GroundPlaneProjector(this.uavCanvas.width, this.uavCanvas.height);
        this.tracker = null;
        
        // Simulation parameters
        this.altitude = 30.0;
        this.pitch = -90.0;
        this.cohesionWeight = 0.50;
        this.maxTimeout = 150;
        this.detectorNoise = 1.5;
        
        // Simulation state variables
        this.isPlaying = false;
        this.speed = 1.0;
        this.currentFrameIdx = 0;
        this.maxFrames = 400;
        this.cows = [];
        this.trees = [];
        this.selectedScenario = "linear";
        
        this.showGroundTruth = true;
        this.showUncertainty = true;
        this.showPaths = true;
        this.showExG = false;
        
        // Active selected track for sidebar profile details
        this.selectedTrackId = 1;
        
        // UAV Camera displacement compensation variables
        this.camOffset = { x: 0, y: 0 };
        this.camAngle = 0.0;

        // Interactive Sandbox state variables
        this.isDrawingPath = false;
        this.drawnPathPoints = [];
        this.draggingTreeIdx = -1;

        // Ground evaluation metrics buffers
        this.history = {
            gt: {}, // frameIdx -> Array of GT [{id, bbox, occluded}]
            pred: {} // frameIdx -> Array of outputs [{trackId, bbox, status}]
        };
        this.motMetrics = {
            totalFrames: 0,
            gtCount: 0,
            tp: 0, fp: 0, fn: 0, idsw: 0,
            visibleGt: 0, visibleRecall: 0,
            occludedGt: 0, occludedRecall: 0,
            occlusionsStarted: 0, occlusionsRecovered: 0
        };

        this.initUI();
        this.resetSimulation();
        this.runLoop();
    }

    initUI() {
        // Play/Pause button
        let playBtn = document.getElementById("btn-play-pause");
        playBtn.onclick = () => {
            this.isPlaying = !this.isPlaying;
            playBtn.innerText = this.isPlaying ? "Pause" : "Play";
            playBtn.classList.toggle("btn-primary", !this.isPlaying);
            playBtn.classList.toggle("btn-secondary", this.isPlaying);
            triggerLog("sys", `Simulation ${this.isPlaying ? 'resumed' : 'paused'}.`);
        };

        // Step button
        document.getElementById("btn-step").onclick = () => {
            this.isPlaying = false;
            playBtn.innerText = "Play";
            playBtn.classList.add("btn-primary");
            playBtn.classList.remove("btn-secondary");
            this.step();
        };

        // Reset button
        document.getElementById("btn-reset").onclick = () => {
            this.resetSimulation();
        };

        // Speed slider
        let speedSlider = document.getElementById("slider-speed");
        let speedVal = document.getElementById("speed-val");
        speedSlider.oninput = () => {
            this.speed = parseFloat(speedSlider.value);
            speedVal.innerText = `${this.speed.toFixed(1)}x`;
        };

        // Toggles
        document.getElementById("chk-ground-truth").onchange = (e) => this.showGroundTruth = e.target.checked;
        document.getElementById("chk-uncertainty").onchange = (e) => this.showUncertainty = e.target.checked;
        document.getElementById("chk-paths").onchange = (e) => this.showPaths = e.target.checked;
        document.getElementById("chk-exg-mask").onchange = (e) => this.showExG = e.target.checked;

        // Clear logs
        document.getElementById("btn-clear-logs").onclick = () => {
            document.getElementById("log-feed").innerHTML = "";
        };

        // Scenarios
        let scenarioBtns = document.querySelectorAll(".btn-scenario");
        scenarioBtns.forEach(btn => {
            btn.onclick = () => {
                scenarioBtns.forEach(b => b.classList.remove("active"));
                btn.classList.add("active");
                this.selectedScenario = btn.dataset.scenario;
                this.resetSimulation();
            };
        });

        // Parameters inputs
        const bindSlider = (id, labelId, prop, suffix = "") => {
            let slider = document.getElementById(id);
            let label = document.getElementById(labelId);
            slider.oninput = () => {
                let v = parseFloat(slider.value);
                this[prop] = v;
                label.innerText = `${v}${suffix}`;
                triggerLog("sys", `Parameter updated: ${prop.toUpperCase()} = ${v}${suffix}`);
            };
        };

        bindSlider("param-altitude", "val-altitude", "altitude", "m");
        bindSlider("param-pitch", "val-pitch", "pitch", "°");
        bindSlider("param-cohesion", "val-cohesion", "cohesionWeight");
        bindSlider("param-timeout", "val-timeout", "maxTimeout", " f");
        bindSlider("param-noise", "val-noise", "detectorNoise", "px");

        // Mouse interaction for dragging trees or drawing path on canvas
        this.uavCanvas.onmousedown = (e) => {
            let rect = this.uavCanvas.getBoundingClientRect();
            let x = (e.clientX - rect.left) * (this.uavCanvas.width / rect.width);
            let y = (e.clientY - rect.top) * (this.uavCanvas.height / rect.height);

            // Check if clicking close to a tree center (within 30px) to drag it
            let clickedTreeIdx = this.trees.findIndex(t => Math.hypot(t.cx - x, t.cy - y) < 30);
            if (clickedTreeIdx !== -1) {
                this.draggingTreeIdx = clickedTreeIdx;
            } else if (this.selectedScenario === "sandbox") {
                // In sandbox, click starts drawing a custom path for Cow 1
                this.isDrawingPath = true;
                this.drawnPathPoints = [[x, y]];
                this.isPlaying = false;
                playBtn.innerText = "Play";
                playBtn.classList.add("btn-primary");
                playBtn.classList.remove("btn-secondary");
            }
        };

        this.uavCanvas.onmousemove = (e) => {
            let rect = this.uavCanvas.getBoundingClientRect();
            let x = (e.clientX - rect.left) * (this.uavCanvas.width / rect.width);
            let y = (e.clientY - rect.top) * (this.uavCanvas.height / rect.height);

            if (this.draggingTreeIdx !== -1) {
                this.trees[this.draggingTreeIdx].cx = x;
                this.trees[this.draggingTreeIdx].cy = y;
                this.generateVegetationSegmentationMask();
            } else if (this.isDrawingPath) {
                this.drawnPathPoints.push([x, y]);
            }
        };

        this.uavCanvas.onmouseup = () => {
            if (this.draggingTreeIdx !== -1) {
                this.draggingTreeIdx = -1;
            }
            if (this.isDrawingPath) {
                this.isDrawingPath = false;
                if (this.drawnPathPoints.length > 5 && this.cows[0]) {
                    // Update cow position to start of drawn path
                    this.cows[0].cx = this.drawnPathPoints[0][0];
                    this.cows[0].cy = this.drawnPathPoints[0][1];
                    // Save custom path
                    this.cows[0].customPath = [...this.drawnPathPoints];
                    this.cows[0].customPathIdx = 0;
                    this.currentFrameIdx = 0;
                    triggerLog("sys", `Drawn custom path for Track 1 (${this.drawnPathPoints.length} points)`);
                }
            }
        };

        // Double click to add a tree in Sandbox
        this.uavCanvas.ondblclick = (e) => {
            if (this.selectedScenario !== "sandbox") return;
            let rect = this.uavCanvas.getBoundingClientRect();
            let x = (e.clientX - rect.left) * (this.uavCanvas.width / rect.width);
            let y = (e.clientY - rect.top) * (this.uavCanvas.height / rect.height);
            this.trees.push({ cx: x, cy: y, r: 80 });
            this.generateVegetationSegmentationMask();
            triggerLog("sys", `Added new vegetation canopy at (${Math.round(x)}, ${Math.round(y)})`);
        };
    }

    resetSimulation() {
        this.currentFrameIdx = 0;
        this.camOffset = { x: 0, y: 0 };
        this.camAngle = 0.0;
        this.history = { gt: {}, pred: {} };
        this.countedSwitches = new Set();
        
        // Reset metrics
        this.motMetrics = {
            totalFrames: 0, gtCount: 0, tp: 0, fp: 0, fn: 0, idsw: 0,
            visibleGt: 0, visibleRecall: 0, occludedGt: 0, occludedRecall: 0,
            occlusionsStarted: 0, occlusionsRecovered: 0
        };
        
        // Set up scenario elements
        Scenarios[this.selectedScenario].init(this);
        
        // Initialize tracker
        this.tracker = new CounterfactualAmodalTracker(this.projector);

        // Pre-run background mask generator
        this.generateVegetationSegmentationMask();

        triggerLog("sys", `Scenario '${Scenarios[this.selectedScenario].title}' initialized.`);
        
        // Select profile Track 1
        this.selectedTrackId = 1;
        updateReIDPanel(null);

        this.updateHUD();
        this.draw();
    }

    generateVegetationSegmentationMask() {
        // Draw the simulated Excess Green index mask
        let ctx = this.maskCtx;
        ctx.fillStyle = "#000000";
        ctx.fillRect(0, 0, this.maskCanvas.width, this.maskCanvas.height);
        
        // Draw trees as solid green on mask canvas
        for (let tree of this.trees) {
            ctx.fillStyle = "rgba(0, 255, 0, 1.0)";
            ctx.beginPath();
            ctx.arc(tree.cx, tree.cy, tree.r, 0, 2 * Math.PI);
            ctx.fill();
        }
    }

    step() {
        if (this.currentFrameIdx >= this.maxFrames) {
            this.isPlaying = false;
            document.getElementById("btn-play-pause").innerText = "Play";
            triggerLog("sys", "Simulation sequence complete.");
            return;
        }

        this.currentFrameIdx++;

        // 1. Move drone camera slightly to test motion compensation (CMC)
        if (this.selectedScenario !== "sandbox") {
            // Slow hover displacement
            this.camOffset.x = 20.0 * Math.sin(this.currentFrameIdx * 0.015);
            this.camOffset.y = 10.0 * Math.cos(this.currentFrameIdx * 0.02);
            
            // Warp tracker states according to frame offsets
            this.tracker.applyCameraMotionCompensation(this.camOffset);
        }

        // 2. Move physical cows
        let visibleDetections = [];
        let detHistograms = [];
        let currentGt = [];

        for (let cow of this.cows) {
            if (cow.updateBehavior) {
                cow.updateBehavior(this.currentFrameIdx);
            }

            // Move cow based on custom drawn sandbox path, or standard velocity
            if (cow.customPath) {
                let idx = Math.min(Math.floor(this.currentFrameIdx * this.speed), cow.customPath.length - 1);
                cow.cx = cow.customPath[idx][0];
                cow.cy = cow.customPath[idx][1];
            } else {
                // If CTRV model scenario, update state dynamically
                if (cow.v !== undefined) {
                    cow.cx += cow.v * Math.cos(cow.theta);
                    cow.cy += cow.v * Math.sin(cow.theta);
                    cow.theta += cow.omega;
                } else {
                    cow.cx += cow.vx;
                    cow.cy += cow.vy;
                }
            }

            // Check if cow center is inside the vegetation mask (occluded)
            let isOccluded = false;
            let mCtx = this.maskCtx;
            let sampleX = Math.floor(cow.cx);
            let sampleY = Math.floor(cow.cy);
            if (sampleX >= 0 && sampleX < this.maskCanvas.width && sampleY >= 0 && sampleY < this.maskCanvas.height) {
                let p = mCtx.getImageData(sampleX, sampleY, 1, 1).data;
                isOccluded = p[1] > 50; // green mask pixel
            }

            // Store ground truth bbox [x1, y1, x2, y2]
            let gtBbox = [cow.cx - cow.w/2, cow.cy - cow.h/2, cow.cx + cow.w/2, cow.cy + cow.h/2];
            currentGt.push({
                id: cow.id,
                bbox: gtBbox,
                isOccluded: isOccluded
            });

            // If visible, detector sees the cow with coordinate noise
            if (!isOccluded) {
                let nx = (Math.random() - 0.5) * this.detectorNoise;
                let ny = (Math.random() - 0.5) * this.detectorNoise;
                let detBbox = [
                    gtBbox[0] - this.camOffset.x + nx,
                    gtBbox[1] - this.camOffset.y + ny,
                    gtBbox[2] - this.camOffset.x + nx,
                    gtBbox[3] - this.camOffset.y + ny
                ];
                visibleDetections.push(detBbox);
                
                // Extract appearance color histogram (simulate HSV embedding)
                let noiseHist = cow.hist.map(v => Math.max(0, v + (Math.random() - 0.5) * 0.05));
                let sum = noiseHist.reduce((s, val) => s + val, 0);
                let normHist = noiseHist.map(v => v / sum);
                detHistograms.push(normHist);
            }

            // Save path point for visual trace
            cow.path.push({ x: cow.cx, y: cow.cy, occluded: isOccluded });
            if (cow.path.length > 200) cow.path.shift();
        }

        // 3. Step the tracker pipeline
        let trackerOutputs = this.tracker.step(
            visibleDetections,
            detHistograms,
            this.maskCanvas,
            this.altitude,
            this.pitch,
            this.maxTimeout,
            this.cohesionWeight,
            this.camOffset
        );

        // 4. Save sequence history for evaluation metrics
        this.history.gt[this.currentFrameIdx] = currentGt;
        this.history.pred[this.currentFrameIdx] = trackerOutputs;

        // 5. Update metrics
        this.evaluateMOT();

        // 6. Draw current scene frame
        this.updateHUD();
        this.draw();
    }

    evaluateMOT() {
        let frameIdx = this.currentFrameIdx;
        let gt = this.history.gt[frameIdx] || [];
        let pred = this.history.pred[frameIdx] || [];

        this.motMetrics.totalFrames++;
        this.motMetrics.gtCount += gt.length;

        // Map tracked targets
        let matchedPredIds = new Set();
        let matchedGtIds = new Set();

        for (let g of gt) {
            if (g.isOccluded) {
                this.motMetrics.occludedGt++;
            } else {
                this.motMetrics.visibleGt++;
            }

            // Find matching output (by IoU)
            let bestIoU = 0.0;
            let matchedOut = null;

            for (let p of pred) {
                let iou = calculateIoU(g.bbox, p.bbox);
                if (iou > bestIoU) {
                    bestIoU = iou;
                    matchedOut = p;
                }
            }

            if (bestIoU > 0.2 && matchedOut) {
                // TP match
                this.motMetrics.tp++;
                matchedGtIds.add(g.id);
                matchedPredIds.add(matchedOut.trackId);

                if (g.isOccluded) {
                    this.motMetrics.occludedRecall++;
                } else {
                    this.motMetrics.visibleRecall++;
                }

                // If this is Cow 1, update active details panel
                if (matchedOut.trackId === this.selectedTrackId) {
                    updateTrackDetailsCard(matchedOut);
                }
            } else {
                // FN match
                this.motMetrics.fn++;
            }
        }

        // False positives
        for (let p of pred) {
            if (!matchedPredIds.has(p.trackId)) {
                this.motMetrics.fp++;
            }
        }

        // Calculate and count occlusion events:
        // An occlusion event starts when a track is first occluded.
        // It recovers successfully when the target re-emerges and matches the same ID.
        for (let tid of Object.keys(this.tracker.amodalAnchors)) {
            let anchor = this.tracker.amodalAnchors[tid];
            if (anchor.framesOccluded === 1) {
                this.motMetrics.occlusionsStarted++;
            }
        }

        // Identify switches: we count new track IDs that exceed the original GT herd size
        if (!this.countedSwitches) this.countedSwitches = new Set();
        let currentActiveIds = this.tracker.baselineTracker.tracks.filter(t => t.timeSinceUpdate === 0).map(t => t.trackId);
        let maxGtId = Math.max(...this.cows.map(c => c.id), 0);
        for (let id of currentActiveIds) {
            if (id > maxGtId && !this.countedSwitches.has(id)) {
                // An ID switch occurred because a track split!
                this.motMetrics.idsw++;
                this.countedSwitches.add(id);
                triggerLog("sys", `[WARNING] ID Switch detected! Tracker spawned a new Track ID: ${id}.`);
            }
        }

        // Update successful recoveries count
        this.motMetrics.occlusionsRecovered = this.motMetrics.occlusionsStarted - Object.keys(this.tracker.amodalAnchors).length;

        // Compile percentages
        let mota = 1.0 - (this.motMetrics.fp + this.motMetrics.fn + this.motMetrics.idsw) / Math.max(1, this.motMetrics.gtCount);
        mota = Math.max(0.0, mota) * 100.0;

        let visRecallRate = (this.motMetrics.visibleRecall / Math.max(1, this.motMetrics.visibleGt)) * 100.0;
        let occRecallRate = (this.motMetrics.occludedRecall / Math.max(1, this.motMetrics.occludedGt)) * 100.0;
        let recoveryRate = (this.motMetrics.occlusionsRecovered / Math.max(1, this.motMetrics.occlusionsStarted)) * 100.0;

        // Inject to UI
        document.getElementById("metric-mota").innerText = `${mota.toFixed(1)}%`;
        document.getElementById("metric-idsw").innerText = `${this.motMetrics.idsw}`;
        document.getElementById("metric-visible-recall").innerText = `${visRecallRate.toFixed(1)}%`;
        document.getElementById("metric-occluded-recall").innerText = `${occRecallRate.toFixed(1)}%`;
        document.getElementById("metric-recovery-rate").innerText = `${recoveryRate.toFixed(1)}%`;
        document.getElementById("metric-recovery-count").innerText = `(${this.motMetrics.occlusionsRecovered}/${this.motMetrics.occlusionsStarted})`;
    }

    updateHUD() {
        document.getElementById("hud-alt").innerText = `${this.altitude.toFixed(1)}m`;
        document.getElementById("hud-pitch").innerText = `${this.pitch.toFixed(1)}°`;
        document.getElementById("hud-roll").innerText = "0.0°";
    }

    draw() {
        let ctx = this.uavCtx;
        
        // 1. Draw pasture grass background with UAV frame displacement warping
        ctx.save();
        ctx.translate(-this.camOffset.x, -this.camOffset.y);
        
        // Draw dark green grass background
        ctx.fillStyle = "#1b3022";
        ctx.fillRect(-200, -200, this.uavCanvas.width + 400, this.uavCanvas.height + 400);

        // Draw dynamic grid lines mapping UAV camera movement
        ctx.strokeStyle = "rgba(255,255,255,0.03)";
        ctx.lineWidth = 1;
        let gridSize = 60;
        for (let x = -200; x < this.uavCanvas.width + 200; x += gridSize) {
            ctx.beginPath(); ctx.moveTo(x, -200); ctx.lineTo(x, this.uavCanvas.height + 200); ctx.stroke();
        }
        for (let y = -200; y < this.uavCanvas.height + 200; y += gridSize) {
            ctx.beginPath(); ctx.moveTo(-200, y); ctx.lineTo(this.uavCanvas.width + 200, y); ctx.stroke();
        }

        // Draw drawing sandbox path if actively drawing
        if (this.isDrawingPath && this.drawnPathPoints.length > 1) {
            ctx.strokeStyle = "rgba(224, 64, 251, 0.4)";
            ctx.lineWidth = 4;
            ctx.lineCap = "round";
            ctx.beginPath();
            ctx.moveTo(this.drawnPathPoints[0][0] + this.camOffset.x, this.drawnPathPoints[0][1] + this.camOffset.y);
            for (let pt of this.drawnPathPoints) {
                ctx.lineTo(pt[0] + this.camOffset.x, pt[1] + this.camOffset.y);
            }
            ctx.stroke();
        }

        // 2. Draw ground truth cattle path traces
        if (this.showPaths) {
            for (let cow of this.cows) {
                if (cow.path.length < 2) continue;
                ctx.lineWidth = 2;
                for (let i = 1; i < cow.path.length; i++) {
                    let p1 = cow.path[i-1];
                    let p2 = cow.path[i];
                    
                    if (p2.occluded) {
                        ctx.strokeStyle = "rgba(255, 167, 38, 0.35)"; // orange dashed trace for occluded EKF prediction
                        ctx.setLineDash([4, 4]);
                    } else {
                        ctx.strokeStyle = "rgba(0, 230, 118, 0.5)"; // green trace for visible tracking
                        ctx.setLineDash([]);
                    }
                    
                    ctx.beginPath();
                    ctx.moveTo(p1.x, p1.y);
                    ctx.lineTo(p2.x, p2.y);
                    ctx.stroke();
                }
                ctx.setLineDash([]);
            }
        }

        // 3. Draw physical cows
        for (let cow of this.cows) {
            let isOccluded = false;
            let currentFrameGt = this.history.gt[this.currentFrameIdx] || [];
            let gtInfo = currentFrameGt.find(g => g.id === cow.id);
            if (gtInfo) isOccluded = gtInfo.isOccluded;

            // Draw cow body shapes
            ctx.save();
            ctx.translate(cow.cx, cow.cy);
            
            // Adjust heading direction angle of cow
            let angle = 0;
            if (cow.vx !== undefined && cow.vy !== undefined) {
                angle = Math.atan2(cow.vy, cow.vx);
            } else if (cow.theta !== undefined) {
                angle = cow.theta;
            }
            ctx.rotate(angle);

            // Set opacity: semi-transparent if occluded under canopy (to see ground truth)
            ctx.globalAlpha = isOccluded ? 0.25 : 1.0;
            
            // Draw Cow silhouette (Holstein spot pattern or brown Swiss)
            ctx.fillStyle = cow.color === "holstein" ? "#ffffff" : (cow.color === "black" ? "#1e1e1e" : "#8d6e63");
            
            // Legs moving animation
            let legWalk = Math.sin(this.currentFrameIdx * 0.2) * 5;
            ctx.strokeStyle = "#1a1a1a";
            ctx.lineWidth = 4;
            // Back legs
            ctx.beginPath(); ctx.moveTo(-15, -12 + legWalk); ctx.lineTo(-15, 12 + legWalk); ctx.stroke();
            // Front legs
            ctx.beginPath(); ctx.moveTo(15, -12 - legWalk); ctx.lineTo(15, 12 - legWalk); ctx.stroke();

            // Body
            ctx.beginPath();
            ctx.ellipse(0, 0, 22, 14, 0, 0, 2 * Math.PI);
            ctx.fill();
            ctx.lineWidth = 1.5;
            ctx.strokeStyle = "#1a1a1a";
            ctx.stroke();

            // Draw spots if Holstein
            if (cow.color === "holstein") {
                ctx.fillStyle = "#111111";
                ctx.beginPath(); ctx.arc(-8, -4, 5, 0, 2*Math.PI); ctx.fill();
                ctx.beginPath(); ctx.arc(4, 5, 6, 0, 2*Math.PI); ctx.fill();
                ctx.beginPath(); ctx.arc(10, -5, 4, 0, 2*Math.PI); ctx.fill();
                ctx.beginPath(); ctx.arc(-14, 2, 4, 0, 2*Math.PI); ctx.fill();
            }

            // Head and ears
            ctx.fillStyle = cow.color === "holstein" ? "#ffffff" : (cow.color === "black" ? "#1a1a1a" : "#8d6e63");
            ctx.beginPath();
            ctx.ellipse(22, 0, 8, 6, 0, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
            
            // Horns / Ears
            ctx.strokeStyle = "#d7ccc8";
            ctx.lineWidth = 2.5;
            ctx.beginPath(); ctx.moveTo(20, -5); ctx.quadraticCurveTo(24, -12, 28, -12); ctx.stroke();
            ctx.beginPath(); ctx.moveTo(20, 5); ctx.quadraticCurveTo(24, 12, 28, 12); ctx.stroke();

            ctx.restore();
            ctx.globalAlpha = 1.0;

            // Draw blue Ground Truth bounding box if toggled
            if (this.showGroundTruth && this.showGroundTruth) {
                ctx.strokeStyle = "rgba(41, 121, 255, 0.4)";
                ctx.lineWidth = 1;
                ctx.setLineDash([2, 2]);
                ctx.strokeRect(cow.cx - cow.w/2, cow.cy - cow.h/2, cow.w, cow.h);
                ctx.setLineDash([]);
                ctx.fillStyle = "rgba(41, 121, 255, 0.8)";
                ctx.font = "9px JetBrains Mono";
                ctx.fillText(`GT:${cow.id}`, cow.cx - cow.w/2, cow.cy - cow.h/2 - 3);
            }
        }

        // 4. Draw trees/vegetation canopy (occlusion zones)
        ctx.globalAlpha = this.showExG ? 0.85 : 0.45;
        for (let tree of this.trees) {
            // Draw canopy shadows first
            ctx.fillStyle = "rgba(10, 18, 14, 0.4)";
            ctx.beginPath();
            ctx.arc(tree.cx + 10, tree.cy + 15, tree.r, 0, 2 * Math.PI);
            ctx.fill();

            // Draw segmented green vegetation canopy (gradient)
            let grad = ctx.createRadialGradient(tree.cx - 20, tree.cy - 20, 10, tree.cx, tree.cy, tree.r);
            if (this.showExG) {
                // Excess Green mask visualizer (neon high contrast green)
                grad.addColorStop(0, "#00e676");
                grad.addColorStop(1, "#1b5e20");
            } else {
                // Organic green canopy visualizer
                grad.addColorStop(0, "#4caf50");
                grad.addColorStop(0.7, "#2e7d32");
                grad.addColorStop(1, "#1b5e20");
            }
            
            ctx.fillStyle = grad;
            ctx.beginPath();
            ctx.arc(tree.cx, tree.cy, tree.r, 0, 2 * Math.PI);
            ctx.fill();
            
            // Draw leafy outlines
            ctx.strokeStyle = "rgba(255,255,255,0.06)";
            ctx.lineWidth = 1.5;
            ctx.stroke();
        }
        ctx.globalAlpha = 1.0;

        // 5. Draw active tracker boxes and amodal anchors
        let currentPreds = this.history.pred[this.currentFrameIdx] || [];
        for (let pred of currentPreds) {
            let [x1, y1, x2, y2] = pred.bbox;
            let w = x2 - x1;
            let h = y2 - y1;

            if (pred.status === "visible") {
                // Visible bounding box (Emerald Green)
                ctx.strokeStyle = varColor("--accent-visible");
                ctx.lineWidth = 2;
                ctx.strokeRect(x1, y1, w, h);
                
                // Track Label
                ctx.fillStyle = varColor("--accent-visible");
                ctx.font = "bold 11px Outfit";
                ctx.fillText(`CATTLE ID: ${pred.trackId}`, x1, y1 - 4);
            } else if (pred.status === "occluded_virtual") {
                // Amodal Anchor predicted bounding box (Orange dotted)
                ctx.strokeStyle = varColor("--accent-occluded");
                ctx.lineWidth = 1.5;
                ctx.setLineDash([4, 3]);
                ctx.strokeRect(x1, y1, w, h);
                ctx.setLineDash([]);
                
                ctx.fillStyle = varColor("--accent-occluded");
                ctx.font = "bold 11px Outfit";
                ctx.fillText(`AMODAL ID: ${pred.trackId} [OCCLUDED]`, x1, y1 - 4);

                // Draw growing uncertainty ellipse Sigma
                if (this.showUncertainty && pred.sigma) {
                    ctx.save();
                    ctx.translate(x1 + w/2, y1 + h/2);
                    
                    // Sigma covariance mapping search bounds (2x2 components)
                    // sxx, sxy, syx, syy. Standard deviation = sqrt(sxx)
                    let stdX = Math.sqrt(pred.sigma[0]);
                    let stdY = Math.sqrt(pred.sigma[3]);
                    
                    // Draw uncertainty search area
                    ctx.strokeStyle = "rgba(255, 167, 38, 0.4)";
                    ctx.lineWidth = 1;
                    ctx.fillStyle = "rgba(255, 167, 38, 0.05)";
                    ctx.beginPath();
                    ctx.ellipse(0, 0, stdX * 5.5, stdY * 5.5, 0, 0, 2 * Math.PI); // Scale up variance factor for display
                    ctx.fill();
                    ctx.stroke();
                    ctx.restore();
                }
            }
        }

        ctx.restore(); // Restore translate offset

        // 6. Draw ground projection top-down view (Bottom-Right Card)
        this.drawGroundProjection();
    }

    drawGroundProjection() {
        let ctx = this.projCtx;
        ctx.fillStyle = "#050709";
        ctx.fillRect(0, 0, this.projCanvas.width, this.projCanvas.height);

        // Draw grid lines mapping 3D perspective warp
        ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
        ctx.lineWidth = 1;
        let cell = 20;
        for (let x = 0; x < this.projCanvas.width; x += cell) {
            ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, this.projCanvas.height); ctx.stroke();
        }
        for (let y = 0; y < this.projCanvas.height; y += cell) {
            ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(this.projCanvas.width, y); ctx.stroke();
        }

        // Draw projected vegetation canopy circles
        let { i2g, g2i } = this.projector.computeHomography(this.altitude, this.pitch, 0.0);
        
        ctx.save();
        // Translate center to grid center
        ctx.translate(this.projCanvas.width / 2, this.projCanvas.height / 2);

        // Render canopy masks in ground space (green circles)
        ctx.fillStyle = "rgba(76, 175, 80, 0.25)";
        ctx.strokeStyle = "rgba(76, 175, 80, 0.5)";
        ctx.lineWidth = 1;
        for (let tree of this.trees) {
            // Map tree image coordinates (tree.cx, tree.cy) to ground coordinates [x_world, y_world]
            let [gx, gy] = this.projector.projectImageToGround(tree.cx - this.camOffset.x, tree.cy - this.camOffset.y, i2g);
            
            // Draw projected tree canopy (scaled)
            let groundScale = 1.6; // scale meters to canvas pixels
            ctx.beginPath();
            ctx.arc(gx * groundScale, gy * groundScale, tree.r * 0.12 * groundScale, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        }

        // Render amodal anchor probability density distribution (2D Gaussian PDF)
        for (let tid of Object.keys(this.tracker.amodalAnchors)) {
            let anchor = this.tracker.amodalAnchors[tid];
            let [gx, gy] = this.projector.projectImageToGround(anchor.cx - this.camOffset.x, anchor.cy - this.camOffset.y, i2g);
            
            let groundScale = 1.6;
            let stdX = Math.sqrt(anchor.Sigma[0]) * 0.12 * groundScale;
            let stdY = Math.sqrt(anchor.Sigma[3]) * 0.12 * groundScale;

            // Draw concentric probability circles showing decaying Gaussian density
            for (let s = 1.0; s <= 3.0; s += 0.8) {
                ctx.fillStyle = `rgba(255, 167, 38, ${0.25 / s})`;
                ctx.beginPath();
                ctx.ellipse(gx * groundScale, gy * groundScale, stdX * s * 3, stdY * s * 3, 0, 0, 2 * Math.PI);
                ctx.fill();
            }
            
            // Draw center dot
            ctx.fillStyle = "#ffa726";
            ctx.beginPath();
            ctx.arc(gx * groundScale, gy * groundScale, 3, 0, 2 * Math.PI);
            ctx.fill();
        }

        // Render current cattle location dots on ground plane
        for (let cow of this.cows) {
            let [gx, gy] = this.projector.projectImageToGround(cow.cx - this.camOffset.x, cow.cy - this.camOffset.y, i2g);
            let groundScale = 1.6;
            ctx.fillStyle = "#2979ff";
            ctx.beginPath();
            ctx.arc(gx * groundScale, gy * groundScale, 2.5, 0, 2 * Math.PI);
            ctx.fill();
        }

        ctx.restore();
    }

    runLoop() {
        let lastTime = performance.now();
        let frameAccumulator = 0.0;
        
        const loop = (time) => {
            let dt = time - lastTime;
            lastTime = time;
            
            if (this.isPlaying) {
                // Accumulate elapsed frames using standard 30 FPS step reference
                frameAccumulator += (dt / 33.33) * this.speed;
                while (frameAccumulator >= 1.0) {
                    this.step();
                    frameAccumulator -= 1.0;
                }
            } else {
                frameAccumulator = 0.0;
            }
            
            // Visual FPS updates
            let fps = 1000.0 / dt;
            document.getElementById("vp-fps").innerText = `FPS: ${Math.min(30.00, fps).toFixed(2)}`;

            requestAnimationFrame(loop);
        };
        
        requestAnimationFrame(loop);
    }
}

// Helper: Get CSS variable color hex
function varColor(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

// --- 9. UI Logger Feed ---
function triggerLog(tag, msg) {
    let feed = document.getElementById("log-feed");
    if (!feed) return;
    
    let now = new Date();
    let timeStr = `${now.getHours().toString().padStart(2,'0')}:${now.getMinutes().toString().padStart(2,'0')}:${now.getSeconds().toString().padStart(2,'0')}.${(now.getMilliseconds()/10).toFixed(0).padStart(2,'0')}`;
    
    let div = document.createElement("div");
    div.className = "log-line";
    div.innerHTML = `
        <span class="log-time">[${timeStr}]</span>
        <span class="log-tag ${tag}">${tag}</span>
        <span class="log-msg">${msg}</span>
    `;
    
    feed.appendChild(div);
    
    // Cap feed items to prevent DOM bloat and slow-downs (especially during automation DOM extraction)
    while (feed.children.length > 200) {
        feed.removeChild(feed.firstChild);
    }
    
    feed.scrollTop = feed.scrollHeight;
}

// --- 10. Side Panel Info cards updates ---
function updateTrackDetailsCard(pred) {
    document.getElementById("reid-track-id").innerText = `Track Profile: ID ${pred.trackId}`;
    
    let speed = Math.hypot(pred.velocity[0], pred.velocity[1]) * 10; // scaled to dm/s
    document.getElementById("reid-track-status").innerHTML = `Status: <span style="color:var(--accent-visible); font-weight:bold;">VISIBLE</span>`;
    document.getElementById("reid-track-kinematics").innerText = `Velocity: ${speed.toFixed(1)} px/s | Uncertainty: 0.0px`;
}

function updateReIDPanel(trackId, anchorHist = null, candidateHist = null, simScore = null) {
    let reidAvatar = document.getElementById("reid-avatar");
    let reidTrackId = document.getElementById("reid-track-id");
    let reidTrackStatus = document.getElementById("reid-track-status");
    let reidTrackKinematics = document.getElementById("reid-track-kinematics");
    
    if (trackId === null) {
        reidAvatar.innerText = "?";
        reidAvatar.className = "profile-avatar";
        reidTrackId.innerText = "Track Profile: None";
        reidTrackStatus.innerText = "Status: No active selection";
        reidTrackKinematics.innerText = "Velocity: - | Uncertainty: -";
        drawHistogram("canvas-hist-anchor", null);
        drawHistogram("canvas-hist-candidate", null);
        document.getElementById("reid-sim-val").innerText = "--";
        document.getElementById("reid-sim-bar").style.width = "0%";
        document.getElementById("reid-sim-bar").className = "gauge-bar-inner";
        return;
    }

    reidAvatar.innerText = `${trackId}`;
    reidAvatar.className = "profile-avatar active";
    reidTrackId.innerText = `Track Profile: ID ${trackId}`;
    reidTrackStatus.innerHTML = `Status: <span style="color:var(--accent-visible); font-weight:bold;">RE-ASSOCIATED</span>`;
    reidTrackKinematics.innerText = `ReID match triggered on emergence!`;

    // Draw histograms
    drawHistogram("canvas-hist-anchor", anchorHist, "#ffa726");
    drawHistogram("canvas-hist-candidate", candidateHist, "#00e676");

    // Similarity score updates
    let pct = Math.round(simScore * 100);
    document.getElementById("reid-sim-val").innerText = `${pct}%`;
    
    let bar = document.getElementById("reid-sim-bar");
    bar.style.width = `${pct}%`;
    bar.className = "gauge-bar-inner";
    if (simScore >= 0.65) {
        bar.classList.add("success");
    } else {
        bar.classList.add("fail");
    }
}

function drawHistogram(canvasId, hist, color = "#ffa726") {
    let canvas = document.getElementById(canvasId);
    let ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!hist) return;

    ctx.fillStyle = color;
    let spacing = 2;
    let w = (canvas.width - spacing * (hist.length - 1)) / hist.length;
    let maxVal = Math.max(...hist, 0.05);

    for (let i = 0; i < hist.length; i++) {
        let val = hist[i];
        let h = (val / maxVal) * (canvas.height - 4);
        ctx.fillRect(i * (w + spacing), canvas.height - h, w, h);
    }
}

// Run engine on DOM loaded
window.onload = () => {
    window.simulator = new OcclusionSimulator();
};
