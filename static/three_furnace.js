/**
 * SmartMelt Studio — 3D Furnace Renderer
 * Coreless induction furnace: coils, refractory lining, molten metal pool,
 * slag layer, solid scrap chunks, flux lumps, all driven by live simulation data.
 */
(function () {
  'use strict';

  let renderer, scene, camera, controls, animFrameId;
  let meshes = {}; // named Three.js meshes updated per-frame
  let scrapPieces = [];
  let fluxLumps = [];
  let isInitialized = false;
  let currentState = { meltedPct: 0, bathTempC: 30, slagKg: 0, undissolvedKg: 0 };
  let animTarget = { meltedPct: 0, bathTempC: 30, slagKg: 0, undissolvedKg: 0 };

  // Color helpers
  function metalColour(TbathC, tapAimC = 1620) {
    const frac = Math.max(0, Math.min(1, (TbathC - 1150) / (tapAimC + 40 - 1150)));
    const r = Math.round(196 + 59 * frac);
    const g = Math.round(46 + 130 * frac);
    const b = Math.round(12 + 26 * frac);
    return new THREE.Color(r / 255, g / 255, b / 255);
  }

  function metalEmissive(TbathC, tapAimC = 1620) {
    const frac = Math.max(0, Math.min(1, (TbathC - 1150) / (tapAimC + 40 - 1150)));
    // Deep red-orange glow that brightens with temperature
    return new THREE.Color(frac * 0.9 + 0.1, frac * 0.3, 0);
  }

  function initScene(container) {
    const W = container.clientWidth || 320;
    const H = container.clientHeight || 240;

    // Renderer
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.appendChild(renderer.domElement);

    // Scene
    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020406);
    scene.fog = new THREE.FogExp2(0x020406, 0.08);

    // Camera
    camera = new THREE.PerspectiveCamera(42, W / H, 0.1, 100);
    camera.position.set(3.5, 2.5, 3.5);
    camera.lookAt(0, 0, 0);

    // Orbit Controls
    if (typeof THREE.OrbitControls !== 'undefined') {
      controls = new THREE.OrbitControls(camera, renderer.domElement);
      controls.enableDamping = true;
      controls.dampingFactor = 0.06;
      controls.maxPolarAngle = Math.PI * 0.78;
      controls.minDistance = 2;
      controls.maxDistance = 9;
      controls.autoRotate = true;
      controls.autoRotateSpeed = 0.4;
    }

    // Lighting
    const ambientLight = new THREE.AmbientLight(0x1a1a2e, 2.0);
    scene.add(ambientLight);

    // Emissive light from molten pool below
    const poolLight = new THREE.PointLight(0xff4400, 0, 2.5);
    poolLight.position.set(0, -0.3, 0);
    poolLight.castShadow = false;
    scene.add(poolLight);
    meshes.poolLight = poolLight;

    // Top rim light
    const rimLight = new THREE.DirectionalLight(0x4fa8d8, 0.6);
    rimLight.position.set(2, 4, 2);
    scene.add(rimLight);

    const backLight = new THREE.DirectionalLight(0xc8802f, 0.3);
    backLight.position.set(-2, 2, -2);
    scene.add(backLight);

    buildFurnaceGeometry();
    isInitialized = true;
    animate();
  }

  function buildFurnaceGeometry() {
    // ── Dimensions (metres, scaled for visual)
    const R_outer = 1.10;  // outer shell radius
    const R_lining = 0.95; // hot-face (inner cavity) radius
    const H_furnace = 1.80; // internal height of cavity
    const H_total = 2.10;
    const COIL_R_INNER = R_outer - 0.01;
    const COIL_R_OUTER = R_outer + 0.14;

    // ── Refractory lining (hollow cylinder, open top)
    const liningShape = new THREE.Shape();
    liningShape.absarc(0, 0, R_outer, 0, Math.PI * 2, false);
    const liningHole = new THREE.Path();
    liningHole.absarc(0, 0, R_lining, 0, Math.PI * 2, true);
    liningShape.holes.push(liningHole);

    const liningExtrudeSettings = { depth: H_total, bevelEnabled: false };
    const liningGeom = new THREE.ExtrudeGeometry(liningShape, liningExtrudeSettings);
    liningGeom.translate(0, -H_total / 2, 0);
    liningGeom.rotateX(-Math.PI / 2);

    const liningMat = new THREE.MeshStandardMaterial({
      color: 0x5a3825,
      roughness: 0.9,
      metalness: 0.05,
      side: THREE.DoubleSide
    });
    const liningMesh = new THREE.Mesh(liningGeom, liningMat);
    scene.add(liningMesh);

    // Bottom cap (refractory floor)
    const floorGeom = new THREE.CircleGeometry(R_lining, 48);
    floorGeom.rotateX(-Math.PI / 2);
    floorGeom.translate(0, -H_total / 2, 0);
    const floorMesh = new THREE.Mesh(floorGeom, liningMat);
    scene.add(floorMesh);

    // Top rim highlight
    const rimGeom = new THREE.TorusGeometry(R_outer, 0.05, 8, 48);
    rimGeom.translate(0, H_total / 2, 0);
    const rimMat = new THREE.MeshStandardMaterial({ color: 0x6b4533, roughness: 0.8 });
    scene.add(new THREE.Mesh(rimGeom, rimMat));

    // ── Induction coils (stacked torus rings)
    const coilMat = new THREE.MeshStandardMaterial({
      color: 0xc8802f,
      roughness: 0.3,
      metalness: 0.85,
      envMapIntensity: 1.5
    });
    const coilGroup = new THREE.Group();
    const nCoils = 10;
    const coilStartY = -H_total / 2 + 0.15;
    const coilSpacing = (H_total - 0.3) / nCoils;

    for (let i = 0; i < nCoils; i++) {
      const coilGeom = new THREE.TorusGeometry(
        (COIL_R_INNER + COIL_R_OUTER) / 2,
        (COIL_R_OUTER - COIL_R_INNER) / 2,
        6, 48
      );
      coilGeom.rotateX(Math.PI / 2);
      const coil = new THREE.Mesh(coilGeom, coilMat);
      coil.position.y = coilStartY + i * coilSpacing;
      coilGroup.add(coil);
    }
    scene.add(coilGroup);

    // ── Cavity (dark inner space, cylinder of air)
    const cavityGeom = new THREE.CylinderGeometry(R_lining - 0.01, R_lining - 0.01, H_total, 48, 1, true);
    const cavityMat = new THREE.MeshBasicMaterial({
      color: 0x030507, side: THREE.BackSide
    });
    scene.add(new THREE.Mesh(cavityGeom, cavityMat));

    // ── Molten steel pool (rises from bottom)
    const metalGeom = new THREE.CylinderGeometry(R_lining - 0.04, R_lining - 0.04, 0.01, 48);
    const metalMat = new THREE.MeshStandardMaterial({
      color: metalColour(30),
      emissive: metalEmissive(30),
      emissiveIntensity: 0.0,
      roughness: 0.15,
      metalness: 0.95
    });
    const metalMesh = new THREE.Mesh(metalGeom, metalMat);
    metalMesh.position.y = -H_total / 2 + 0.005;
    scene.add(metalMesh);
    meshes.metal = metalMesh;
    meshes.metalMat = metalMat;

    // Bright surface shimmer on top of pool
    const surfaceGeom = new THREE.CircleGeometry(R_lining - 0.05, 48);
    surfaceGeom.rotateX(-Math.PI / 2);
    const surfaceMat = new THREE.MeshStandardMaterial({
      color: 0xffd166,
      emissive: 0xffa040,
      emissiveIntensity: 0.0,
      transparent: true,
      opacity: 0,
      roughness: 0.05,
      metalness: 0.99
    });
    const surfaceMesh = new THREE.Mesh(surfaceGeom, surfaceMat);
    surfaceMesh.position.y = -H_total / 2 + 0.01;
    scene.add(surfaceMesh);
    meshes.surface = surfaceMesh;
    meshes.surfaceMat = surfaceMat;

    // ── Slag layer (sits on top of metal, khaki/brown)
    const slagGeom = new THREE.CylinderGeometry(R_lining - 0.04, R_lining - 0.04, 0.001, 48);
    const slagMat = new THREE.MeshStandardMaterial({
      color: 0x7d6b48,
      roughness: 0.85,
      metalness: 0.0
    });
    const slagMesh = new THREE.Mesh(slagGeom, slagMat);
    slagMesh.visible = false;
    scene.add(slagMesh);
    meshes.slag = slagMesh;

    // ── Solid scrap: randomised polyhedra chunks above pool
    scrapPieces = [];
    const scrapMat = new THREE.MeshStandardMaterial({
      color: 0x8792a0, roughness: 0.85, metalness: 0.4
    });
    const rng = mulberry32(7);
    for (let i = 0; i < 22; i++) {
      const geomType = Math.floor(rng() * 3);
      let geom;
      if (geomType === 0) geom = new THREE.BoxGeometry(rng() * 0.15 + 0.06, rng() * 0.10 + 0.04, rng() * 0.12 + 0.05);
      else if (geomType === 1) geom = new THREE.TetrahedronGeometry(rng() * 0.08 + 0.04);
      else geom = new THREE.OctahedronGeometry(rng() * 0.07 + 0.03);

      const px = (rng() - 0.5) * (R_lining - 0.25) * 1.6;
      const pz = (rng() - 0.5) * (R_lining - 0.25) * 1.6;
      // Keep within radius
      const dist = Math.sqrt(px * px + pz * pz);
      const maxR = R_lining - 0.18;
      const sx = dist > maxR ? px * maxR / dist : px;
      const sz = dist > maxR ? pz * maxR / dist : pz;
      const py = -H_total / 2 + (rng() * H_furnace * 0.5) + 0.05;

      const mesh = new THREE.Mesh(geom, scrapMat.clone());
      mesh.position.set(sx, py, sz);
      mesh.rotation.set(rng() * Math.PI, rng() * Math.PI, rng() * Math.PI);
      mesh.castShadow = true;
      scene.add(mesh);
      scrapPieces.push({ mesh, baseY: py, baseScale: 1 });
    }

    // ── Flux lumps: off-white spheres
    fluxLumps = [];
    const fluxMat = new THREE.MeshStandardMaterial({
      color: 0xece6d4, roughness: 0.75, metalness: 0.0
    });
    const rng2 = mulberry32(42);
    for (let i = 0; i < 8; i++) {
      const r = rng2() * 0.06 + 0.025;
      const geom = new THREE.SphereGeometry(r, 8, 8);
      const px = (rng2() - 0.5) * (R_lining - 0.25) * 1.4;
      const pz = (rng2() - 0.5) * (R_lining - 0.25) * 1.4;
      const dist = Math.sqrt(px * px + pz * pz);
      const maxR = R_lining - 0.22;
      const sx = dist > maxR ? px * maxR / dist : px;
      const sz = dist > maxR ? pz * maxR / dist : pz;
      const mesh = new THREE.Mesh(geom, fluxMat.clone());
      mesh.position.set(sx, -H_total / 2 + 0.05, sz);
      mesh.visible = false;
      scene.add(mesh);
      fluxLumps.push({ mesh, baseR: r });
    }

    // Store layout constants for updates
    meshes.R_lining = R_lining;
    meshes.H_total = H_total;
    meshes.H_furnace = H_furnace;
  }

  // Simple seeded PRNG (Mulberry32) for deterministic scrap placement
  function mulberry32(seed) {
    let s = seed;
    return function () {
      s |= 0; s = s + 0x6D2B79F5 | 0;
      let t = Math.imul(s ^ s >>> 15, 1 | s);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  /**
   * Update furnace 3D state from simulation snapshot values.
   * @param {number} meltedPct  0–100
   * @param {number} TbathC     bath temperature in °C
   * @param {number} slagKg     total slag mass kg
   * @param {number} undissolvedKg  undissolved additions kg
   * @param {number} tapAimC    tap temperature aim
   */
  function updateFurnaceState(meltedPct, TbathC, slagKg, undissolvedKg, tapAimC = 1620) {
    animTarget = { meltedPct, bathTempC: TbathC, slagKg, undissolvedKg };
    // Update badge
    const badge = document.getElementById('furnace-temp-badge');
    if (badge) badge.textContent = `${TbathC.toFixed(0)} °C`;
  }

  function lerpState(dt) {
    const alpha = 1 - Math.pow(0.02, dt);
    currentState.meltedPct += (animTarget.meltedPct - currentState.meltedPct) * alpha;
    currentState.bathTempC += (animTarget.bathTempC - currentState.bathTempC) * alpha;
    currentState.slagKg += (animTarget.slagKg - currentState.slagKg) * alpha;
    currentState.undissolvedKg += (animTarget.undissolvedKg - currentState.undissolvedKg) * alpha;
  }

  let lastTime = 0;
  function animate(time = 0) {
    animFrameId = requestAnimationFrame(animate);
    const dt = Math.min((time - lastTime) / 1000, 0.1);
    lastTime = time;

    if (!isInitialized) return;

    lerpState(dt);
    updateMeshes(dt);

    if (controls) controls.update();
    renderer.render(scene, camera);
  }

  function updateMeshes(dt) {
    const { meltedPct, bathTempC, slagKg, undissolvedKg } = currentState;
    const { R_lining, H_total, H_furnace } = meshes;
    const tapAimC = 1620;

    const melted = Math.max(0, Math.min(1, meltedPct / 100));
    const usable = H_furnace * 0.92;
    const liqH = usable * melted;
    const liqBottomY = -H_total / 2;
    const liqTopY = liqBottomY + liqH;

    const slagH = slagKg > 0 && liqH > 0.05 ? Math.min(0.12, 0.04 + slagKg / 700) : 0;
    const slagTopY = liqTopY + slagH;

    // ── Metal pool scale and position
    if (meshes.metal) {
      const metalMesh = meshes.metal;
      if (liqH > 0.01) {
        metalMesh.visible = true;
        metalMesh.scale.y = Math.max(liqH, 0.01) * 100;  // CylinderGeometry height=0.01
        metalMesh.position.y = liqBottomY + liqH / 2;
      } else {
        metalMesh.visible = false;
      }

      const col = metalColour(bathTempC, tapAimC);
      const emv = metalEmissive(bathTempC, tapAimC);
      const frac = Math.max(0, Math.min(1, (bathTempC - 1150) / (tapAimC + 40 - 1150)));
      meshes.metalMat.color.copy(col);
      meshes.metalMat.emissive.copy(emv);
      meshes.metalMat.emissiveIntensity = frac * 1.8;

      // Pool glow light
      if (meshes.poolLight) {
        meshes.poolLight.intensity = frac * 3.5;
        meshes.poolLight.position.y = liqTopY - 0.1;
        meshes.poolLight.color.set(col);
      }
    }

    // ── Surface shimmer
    if (meshes.surface && meshes.surfaceMat) {
      const frac = Math.max(0, Math.min(1, (bathTempC - 1150) / (tapAimC + 40 - 1150)));
      meshes.surface.position.y = liqTopY + 0.005;
      meshes.surface.visible = liqH > 0.01;
      meshes.surfaceMat.opacity = liqH > 0.05 ? frac * 0.85 : 0;
      meshes.surfaceMat.emissiveIntensity = frac * 2.0;
    }

    // ── Slag
    if (meshes.slag) {
      if (slagH > 0 && liqH > 0.05) {
        meshes.slag.visible = true;
        meshes.slag.scale.y = slagH * 1000;
        meshes.slag.position.y = liqTopY + slagH / 2;
      } else {
        meshes.slag.visible = false;
      }
    }

    // ── Scrap pieces shrink as melted increases
    const solidFrac = 1 - melted;
    const nVisibleScrap = Math.floor(solidFrac * scrapPieces.length);
    scrapPieces.forEach((item, i) => {
      if (i < nVisibleScrap) {
        item.mesh.visible = true;
        // Scale down near meltdown
        const scaleF = solidFrac < 0.1 ? solidFrac * 10 : 1;
        item.mesh.scale.setScalar(scaleF);
        // Keep pieces above pool level
        const newY = Math.max(item.baseY, liqTopY + 0.04);
        item.mesh.position.y = newY;
        // Slow rotation animation for "tumbling in bath"
        item.mesh.rotation.y += dt * 0.3 * (i % 2 === 0 ? 1 : -1);
      } else {
        item.mesh.visible = false;
      }
    });

    // ── Flux lumps visible when undissolved additions > 1 kg and pool exists
    const nFluxVisible = undissolvedKg > 1 && liqH > 0.08
      ? Math.min(Math.floor(1 + undissolvedKg / 12), fluxLumps.length)
      : 0;
    fluxLumps.forEach((item, i) => {
      if (i < nFluxVisible) {
        item.mesh.visible = true;
        item.mesh.position.y = liqTopY - 0.02 + Math.sin(lastTime * 0.001 + i * 1.3) * 0.015;
      } else {
        item.mesh.visible = false;
      }
    });
  }

  // Handle resize
  function onResize() {
    if (!renderer || !camera) return;
    const container = renderer.domElement.parentElement;
    if (!container) return;
    const W = container.clientWidth;
    const H = container.clientHeight;
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    renderer.setSize(W, H);
  }

  // Public API
  window.ThreeFurnace = {
    init(containerId) {
      const container = document.getElementById(containerId);
      if (!container || isInitialized) return;
      initScene(container);
      window.addEventListener('resize', onResize);
    },
    update(meltedPct, TbathC, slagKg, undissolvedKg, tapAimC) {
      if (!isInitialized) return;
      updateFurnaceState(meltedPct, TbathC, slagKg, undissolvedKg, tapAimC);
    },
    reset() {
      updateFurnaceState(0, 30, 0, 0, 1620);
    }
  };
})();
