/**
 * Engram Web Gateway — Request Signer
 *
 * Signs outgoing miner requests with an sr25519 hotkey so miners running
 * with REQUIRE_HOTKEY_SIG=true can verify the web gateway's identity.
 *
 * Config
 * ------
 * GATEWAY_MNEMONIC  — BIP-39 mnemonic (or Substrate derivation path like //Alice).
 *                     When unset, requests are forwarded unsigned (backward compat).
 *
 * Signing protocol (matches engram/miner/auth.py)
 * -----------------------------------------------
 * canonical_message = `${nonce}:${endpoint}:${bodyHash}`
 * bodyHash          = SHA-256(JSON.stringify(payload, sorted keys), hex)
 * payload           = body fields excluding "hotkey", "nonce", "signature"
 * nonce             = Date.now()  (unix ms — ±30s replay window on miner)
 */

import { createHash } from "crypto";

// Lazy-init the keypair once — avoids WASM startup cost on every request.
let _signerCache: { address: string; sign: (msg: Uint8Array) => Uint8Array } | null = null;
let _signerInitialised = false;

async function getSigner() {
  if (_signerInitialised) return _signerCache;
  _signerInitialised = true;

  const mnemonic = process.env.GATEWAY_MNEMONIC;
  if (!mnemonic) return null;

  try {
    const { Keyring } = await import("@polkadot/keyring");
    const { cryptoWaitReady } = await import("@polkadot/util-crypto");
    await cryptoWaitReady();
    const keyring = new Keyring({ type: "sr25519" });
    const pair = keyring.addFromUri(mnemonic);
    _signerCache = { address: pair.address, sign: (msg) => pair.sign(msg) };
  } catch (err) {
    console.error("[gateway-signer] Failed to initialise keypair:", err);
  }

  return _signerCache;
}

function payloadHash(body: Record<string, unknown>): string {
  const payload: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(body)) {
    if (k !== "hotkey" && k !== "nonce" && k !== "signature") payload[k] = v;
  }
  const serialised = JSON.stringify(payload, Object.keys(payload).sort());
  return createHash("sha256").update(serialised).digest("hex");
}

/**
 * Add hotkey/nonce/signature fields to a miner request body.
 * Returns the original body unchanged when GATEWAY_MNEMONIC is not set.
 */
export async function signMinerRequest(
  body: Record<string, unknown>,
  endpoint: "IngestSynapse" | "QuerySynapse" | "ChallengeSynapse"
): Promise<Record<string, unknown>> {
  const signer = await getSigner();
  if (!signer) return body; // unsigned — miner must have REQUIRE_HOTKEY_SIG=false

  const nonce = Date.now();
  const bodyWithMeta = { ...body, hotkey: signer.address, nonce };
  const bh = payloadHash(bodyWithMeta);
  const message = new TextEncoder().encode(`${nonce}:${endpoint}:${bh}`);
  const sigBytes = signer.sign(message);
  const sigHex = "0x" + Buffer.from(sigBytes).toString("hex");

  return { ...bodyWithMeta, signature: sigHex };
}

/** Returns the gateway hotkey SS58 address, or null when unconfigured. */
export async function gatewayHotkey(): Promise<string | null> {
  const signer = await getSigner();
  return signer?.address ?? null;
}
