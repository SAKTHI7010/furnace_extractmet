/**
 * SmartMelt Studio — 3D Furnace Renderer
 * Coreless induction furnace: coils, refractory lining, molten metal pool,
 * slag layer, solid scrap chunks, flux lumps, all driven by live simulation data.
 *
 * Fixes (vs original):
 *  - WebGL availability check before renderer creation — throws friendly error
 *  - getBoundingClientRect() for container size (clientWidth/Height can be 0)
 *  - ResizeObserver to keep canvas in sync with container resizes
 *  - All public API methods guarded against uninitialised renderer
 *  - slag fill uses full slagKg (all species), matching engine slag_total_kg
 */
(function () {
  'use strict';

  let renderer, scene, camera, controls, animFrameId;
  let meshes = {};
  let scrapPieces = [];
  let fluxLumps = [];
  let isInitialized = false;
  let currentState = { meltedPct: 0, bathTempC: 30, slagKg: 0, undissolvedKg: 0 };
  let animTarget   = { meltedPct: 0, bathTempC: 30, slagKg: 0, undissolvedKg: 0 };

  // WebGL detection
  function isWebGLAvailable() {
    try {
      const canvas = document.createElement('canvas');
      return !!(window.WebGLRenderingContext &&
        (canvas.getContext('webgl') || canvas.getContext('experimental-webgl')));
    } catch (e) { return false; }
  }

  // Color helpers
  function metalColour(TbathC, tapAimC) {
    tapAimC = tapAimC || 1620;
    const frac = Math.max(0, Math.min(1, (TbathC - 1150) / (tapAimC + 40 - 1150)));
    return new THREE.Color(
      (196 + 59 * frac) / 255,
      (46  + 130* frac) / 255,
      (12  + 26 * frac) / 255
    );
  }

  function metalEmissive(TbathC, tapAimC) {
    tapAimC = tapAimC || 1620;
    const frac = Math.max(0, Math.min(1, (TbathC - 1150) / (tapAimC + 40 - 1150)));
    return new THREE.Color(frac * 0.9 + 0.1, frac * 0.3, 0);
  }

  function initScene(container) {
    // getBoundingClientRect is reliable even before first paint
    var rect = container.getBoundingClientRect();
    var W = rect.width  || container.clientWidth  || 320;
    var H = rect.height || container.clientHeight || 260;

    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(W, H);
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.2;
    container.appendChild(renderer.domElement);

    scene = new THREE.Scene();
    scene.background = new THREE.Color(0x020406);
    scene.fog = new THREE.FogExp2(0x020406, 0.08);

    camera = new THREE.PerspectiveCamera(42, W / H, 0.1, 100);
    camera.position.set(3.5, 2.5, 3.5);
    camera.lookAt(0, 0, 0);

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

    scene.add(new THREE.AmbientLight(0x1a1a2e, 2.0));

    var poolLight = new THREE.PointLight(0xff4400, 0, 2.5);
    poolLight.position.set(0, -0.3, 0);
    scene.add(poolLight);
    meshes.poolLight = poolLight;

    var rimLight = new THREE.DirectionalLight(0x4fa8d8, 0.6);
    rimLight.position.set(2, 4, 2);
    scene.add(rimLight);

    var backLight = new THREE.DirectionalLight(0xc8802f, 0.3);
    backLight.position.set(-2, 2, -2);
    scene.add(backLight);

    buildFurnaceGeometry();
    isInitialized = true;

    // ResizeObserver keeps canvas in sync when container resizes
    if (typeof ResizeObserver !== 'undefined') {
      var ro = new ResizeObserver(function() { onResize(container); });
      ro.observe(container);
    }
    window.addEventListener('resize', function() { onResize(container); });

    animate();
  }

  function buildFurnaceGeometry() {
    var R_outer  = 1.10;
    var R_lining = 0.95;
    var H_furnace = 1.80;
    var H_total   = 2.10;
    var COIL_R_INNER = R_outer - 0.01;
    var COIL_R_OUTER = R_outer + 0.14;

    var liningShape = new THREE.Shape();
    liningShape.absarc(0, 0, R_outer, 0, Math.PI * 2, false);
    var liningHole = new THREE.Path();
    liningHole.absarc(0, 0, R_lining, 0, Math.PI * 2, true);
    liningShape.holes.push(liningHole);

    var liningGeom = new THREE.ExtrudeGeometry(liningShape, { depth: H_total, bevelEnabled: false });
    liningGeom.translate(0, -H_total / 2, 0);
    liningGeom.rotateX(-Math.PI / 2);

    var liningMat = new THREE.MeshStandardMaterial({
      color: 0x5a3825, roughness: 0.9, metalness: 0.05, side: THREE.DoubleSide
    });
    scene.add(new THREE.Mesh(liningGeom, liningMat));

    var floorGeom = new THREE.CircleGeometry(R_lining, 48);
    floorGeom.rotateX(-Math.PI / 2);
    floorGeom.translate(0, -H_total / 2, 0);
    scene.add(new THREE.Mesh(floorGeom, liningMat));

    var rimGeom = new THREE.TorusGeometry(R_outer, 0.05, 8, 48);
    rimGeom.translate(0, H_total / 2, 0);
    scene.add(new THREE.Mesh(rimGeom, new THREE.MeshStandardMaterial({ color: 0x6b4533, roughness: 0.8 })));

    var coilMat = new THREE.MeshStandardMaterial({ color: 0xc8802f, roughness: 0.3, metalness: 0.85 });
    var coilGroup = new THREE.Group();
    var nCoils = 10;
    var coilStartY  = -H_total / 2 + 0.15;
    var coilSpacing = (H_total - 0.3) / nCoils;
    for (var i = 0; i < nCoils; i++) {
      var cg = new THREE.TorusGeometry(
        (COIL_R_INNER + COIL_R_OUTER) / 2,
        (COIL_R_OUTER - COIL_R_INNER) / 2, 6, 48
      );
      cg.rotateX(Math.PI / 2);
      var coil = new THREE.Mesh(cg, coilMat);
      coil.position.y = coilStartY + i * coilSpacing;
      coilGroup.add(coil);
    }
    scene.add(coilGroup);

    var cavityGeom = new THREE.CylinderGeometry(R_lining - 0.01, R_lining - 0.01, H_total, 48, 1, true);
    scene.add(new THREE.Mesh(cavityGeom, new THREE.MeshBasicMaterial({ color: 0x030507, side: THREE.BackSide })));

    var metalGeom = new THREE.CylinderGeometry(R_lining - 0.04, R_lining - 0.04, 0.01, 48);
    var metalMat  = new THREE.MeshStandardMaterial({
      color: metalColour(30), emissive: metalEmissive(30),
      emissiveIntensity: 0, roughness: 0.15, metalness: 0.95
    });
    var metalMesh = new THREE.Mesh(metalGeom, metalMat);
    metalMesh.position.y = -H_total / 2 + 0.005;
    scene.add(metalMesh);
    meshes.metal    = metalMesh;
    meshes.metalMat = metalMat;

    var surfaceGeom = new THREE.CircleGeometry(R_lining - 0.05, 48);
    surfaceGeom.rotateX(-Math.PI / 2);
    var surfaceMat = new THREE.MeshStandardMaterial({
      color: 0xffd166, emissive: 0xffa040, emissiveIntensity: 0,
      transparent: true, opacity: 0, roughness: 0.05, metalness: 0.99
    });
    var surfaceMesh = new THREE.Mesh(surfaceGeom, surfaceMat);
    surfaceMesh.position.y = -H_total / 2 + 0.01;
    scene.add(surfaceMesh);
    meshes.surface    = surfaceMesh;
    meshes.surfaceMat = surfaceMat;

    var slagGeom = new THREE.CylinderGeometry(R_lining - 0.04, R_lining - 0.04, 0.001, 48);
    var slagMesh = new THREE.Mesh(slagGeom, new THREE.MeshStandardMaterial({
      color: 0x7d6b48, roughness: 0.85, metalness: 0
    }));
    slagMesh.visible = false;
    scene.add(slagMesh);
    meshes.slag = slagMesh;

    scrapPieces = [];
    var scrapMat = new THREE.MeshStandardMaterial({ color: 0x8792a0, roughness: 0.85, metalness: 0.4 });
    var rng = mulberry32(7);
    for (var j = 0; j < 22; j++) {
      var gt = Math.floor(rng() * 3);
      var geom;
      if (gt === 0)      geom = new THREE.BoxGeometry(rng()*0.15+0.06, rng()*0.10+0.04, rng()*0.12+0.05);
      else if (gt === 1) geom = new THREE.TetrahedronGeometry(rng()*0.08+0.04);
      else               geom = new THREE.OctahedronGeometry(rng()*0.07+0.03);
      var px = (rng() - 0.5) * (R_lining - 0.25) * 1.6;
      var pz = (rng() - 0.5) * (R_lining - 0.25) * 1.6;
      var dist = Math.sqrt(px*px + pz*pz);
      var maxR = R_lining - 0.18;
      var sx = dist > maxR ? px*maxR/dist : px;
      var sz = dist > maxR ? pz*maxR/dist : pz;
      var py = -H_total / 2 + (rng() * H_furnace * 0.5) + 0.05;
      var sm = new THREE.Mesh(geom, scrapMat.clone());
      sm.position.set(sx, py, sz);
      sm.rotation.set(rng()*Math.PI, rng()*Math.PI, rng()*Math.PI);
      scene.add(sm);
      scrapPieces.push({ mesh: sm, baseY: py });
    }

    fluxLumps = [];
    var fluxMat = new THREE.MeshStandardMaterial({ color: 0xece6d4, roughness: 0.75, metalness: 0 });
    var rng2 = mulberry32(42);
    for (var k = 0; k < 8; k++) {
      var fr = rng2()*0.06 + 0.025;
      var fg = new THREE.SphereGeometry(fr, 8, 8);
      var fpx = (rng2() - 0.5) * (R_lining - 0.25) * 1.4;
      var fpz = (rng2() - 0.5) * (R_lining - 0.25) * 1.4;
      var fdist = Math.sqrt(fpx*fpx + fpz*fpz);
      var fmaxR = R_lining - 0.22;
      var fm = new THREE.Mesh(fg, fluxMat.clone());
      fm.position.set(fdist > fmaxR ? fpx*fmaxR/fdist : fpx, -H_total/2+0.05, fdist > fmaxR ? fpz*fmaxR/fdist : fpz);
      fm.visible = false;
      scene.add(fm);
      fluxLumps.push({ mesh: fm });
    }

    meshes.R_lining  = R_lining;
    meshes.H_total   = H_total;
    meshes.H_furnace = H_furnace;
  }

  function mulberry32(seed) {
    var s = seed;
    return function () {
      s |= 0; s = s + 0x6D2B79F5 | 0;
      var t = Math.imul(s ^ s >>> 15, 1 | s);
      t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
      return ((t ^ t >>> 14) >>> 0) / 4294967296;
    };
  }

  function updateFurnaceState(meltedPct, TbathC, slagKg, undissolvedKg, tapAimC) {
    animTarget = { meltedPct: meltedPct, bathTempC: TbathC, slagKg: slagKg, undissolvedKg: undissolvedKg };
    var badge = document.getElementById('furnace-temp-badge');
    if (badge) badge.textContent = (TbathC || 0).toFixed(0) + ' \u00b0C';
  }

  function lerpState(dt) {
    var a = 1 - Math.pow(0.02, dt);
    currentState.meltedPct     += (animTarget.meltedPct     - currentState.meltedPct)     * a;
    currentState.bathTempC     += (animTarget.bathTempC     - currentState.bathTempC)     * a;
    currentState.slagKg        += (animTarget.slagKg        - currentState.slagKg)        * a;
    currentState.undissolvedKg += (animTarget.undissolvedKg - currentState.undissolvedKg) * a;
  }

  var lastTime = 0;
  function animate(time) {
    time = time || 0;
    animFrameId = requestAnimationFrame(animate);
    var dt = Math.min((time - lastTime) / 1000, 0.1);
    lastTime = time;
    if (!isInitialized || !renderer) return;
    lerpState(dt);
    updateMeshes(dt);
    if (controls) controls.update();
    renderer.render(scene, camera);
  }

  function updateMeshes(dt) {
    var meltedPct     = currentState.meltedPct;
    var bathTempC     = currentState.bathTempC;
    var slagKg        = currentState.slagKg;
    var undissolvedKg = currentState.undissolvedKg;
    var R_lining  = meshes.R_lining;
    var H_total   = meshes.H_total;
    var H_furnace = meshes.H_furnace;
    var tapAimC   = 1620;

    var melted     = Math.max(0, Math.min(1, meltedPct / 100));
    var liqH       = H_furnace * 0.92 * melted;
    var liqBottomY = -H_total / 2;
    var liqTopY    = liqBottomY + liqH;
    var slagH      = (slagKg > 0 && liqH > 0.05) ? Math.min(0.12, 0.04 + slagKg / 700) : 0;
    var frac       = Math.max(0, Math.min(1, (bathTempC - 1150) / (tapAimC + 40 - 1150)));

    if (meshes.metal) {
      if (liqH > 0.01) {
        meshes.metal.visible  = true;
        meshes.metal.scale.y  = Math.max(liqH, 0.01) * 100;
        meshes.metal.position.y = liqBottomY + liqH / 2;
      } else {
        meshes.metal.visible = false;
      }
      meshes.metalMat.color.copy(metalColour(bathTempC, tapAimC));
      meshes.metalMat.emissive.copy(metalEmissive(bathTempC, tapAimC));
      meshes.metalMat.emissiveIntensity = frac * 1.8;
      if (meshes.poolLight) {
        meshes.poolLight.intensity  = frac * 3.5;
        meshes.poolLight.position.y = liqTopY - 0.1;
        meshes.poolLight.color.set(metalColour(bathTempC, tapAimC));
      }
    }

    if (meshes.surface && meshes.surfaceMat) {
      meshes.surface.position.y = liqTopY + 0.005;
      meshes.surface.visible    = liqH > 0.01;
      meshes.surfaceMat.opacity = liqH > 0.05 ? frac * 0.85 : 0;
      meshes.surfaceMat.emissiveIntensity = frac * 2.0;
    }

    if (meshes.slag) {
      if (slagH > 0 && liqH > 0.05) {
        meshes.slag.visible   = true;
        meshes.slag.scale.y   = slagH * 1000;
        meshes.slag.position.y = liqTopY + slagH / 2;
      } else {
        meshes.slag.visible = false;
      }
    }

    var solidFrac     = 1 - melted;
    var nVisibleScrap = Math.floor(solidFrac * scrapPieces.length);
    scrapPieces.forEach(function(item, i) {
      if (i < nVisibleScrap) {
        item.mesh.visible = true;
        item.mesh.scale.setScalar(solidFrac < 0.1 ? solidFrac * 10 : 1);
        item.mesh.position.y = Math.max(item.baseY, liqTopY + 0.04);
        item.mesh.rotation.y += dt * 0.3 * (i % 2 === 0 ? 1 : -1);
      } else {
        item.mesh.visible = false;
      }
    });

    var nFlux = (undissolvedKg > 1 && liqH > 0.08)
      ? Math.min(Math.floor(1 + undissolvedKg / 12), fluxLumps.length) : 0;
    fluxLumps.forEach(function(item, i) {
      item.mesh.visible = i < nFlux;
      if (i < nFlux) item.mesh.position.y = liqTopY - 0.02 + Math.sin(lastTime * 0.001 + i * 1.3) * 0.015;
    });
  }

  function onResize(container) {
    if (!renderer || !camera) return;
    var c = container || (renderer.domElement && renderer.domElement.parentElement);
    if (!c) return;
    var rect = c.getBoundingClientRect();
    var W = rect.width  || c.clientWidth  || 320;
    var H = rect.height || c.clientHeight || 260;
    if (W < 10 || H < 10) return;
    camera.aspect = W / H;
    camera.updateProjectionMatrix();
    renderer.setSize(W, H);
  }

  // Public API — all methods guarded against uninitialised state
  window.ThreeFurnace = {
    init: function(containerId) {
      if (isInitialized) return;
      var container = document.getElementById(containerId);
      if (!container) { console.warn('[ThreeFurnace] Container not found:', containerId); return; }
      if (!isWebGLAvailable()) throw new Error('WebGL not supported by this browser');
      initScene(container);
    },
    update: function(meltedPct, TbathC, slagKg, undissolvedKg, tapAimC) {
      var badge = document.getElementById('furnace-temp-badge');
      if (badge && TbathC != null) badge.textContent = (+TbathC).toFixed(0) + ' \u00b0C';
      if (!isInitialized || !renderer) return;
      updateFurnaceState(meltedPct, TbathC, slagKg, undissolvedKg, tapAimC);
    },
    reset: function() {
      var badge = document.getElementById('furnace-temp-badge');
      if (badge) badge.textContent = '30 \u00b0C';
      if (!isInitialized || !renderer) return;
      updateFurnaceState(0, 30, 0, 0, 1620);
    }
  };

})();
