import React, { useState, useEffect } from "react";
import axios from "axios";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Link2, Shield, Clock, FileText } from "lucide-react";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

const Blockchain = () => {
  const [blocks, setBlocks] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchBlockchain = async () => {
      try {
        const response = await axios.get(`${API}/blockchain`);
        setBlocks(response.data);
        setLoading(false);
      } catch (error) {
        console.error("Error fetching blockchain:", error);
        setLoading(false);
      }
    };

    fetchBlockchain();
    const interval = setInterval(fetchBlockchain, 10000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="space-y-6" data-testid="blockchain-page">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-4xl font-bold bg-gradient-to-r from-purple-400 to-blue-400 bg-clip-text text-transparent">
            Blockchain Ledger
          </h1>
          <p className="text-slate-400 mt-2">Immutable evidence chain for all detections</p>
        </div>
        <div className="bg-purple-500/10 border border-purple-500/30 rounded-lg px-4 py-2">
          <p className="text-sm text-slate-300">
            Total Blocks: <span className="font-bold text-purple-400">{blocks.length}</span>
          </p>
        </div>
      </div>

      {loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-pulse text-slate-400">Loading blockchain...</div>
        </div>
      ) : blocks.length === 0 ? (
        <Card className="bg-slate-800/50 border-slate-700">
          <CardContent className="py-12">
            <div className="text-center">
              <Link2 className="h-20 w-20 mx-auto text-slate-600 mb-4" />
              <h3 className="text-xl font-semibold text-slate-300 mb-2">No blocks yet</h3>
              <p className="text-slate-400">Blockchain will start recording when first detection occurs</p>
            </div>
          </CardContent>
        </Card>
      ) : (
        <div className="space-y-4">
          {blocks.map((block, index) => (
            <Card
              key={block.block_id}
              data-testid={`block-${index}`}
              className="bg-slate-800/50 border-slate-700 hover:bg-slate-800/70 transition-all"
            >
              <CardHeader>
                <div className="flex items-center justify-between">
                  <CardTitle className="text-white flex items-center">
                    <Link2 className="h-5 w-5 mr-2 text-purple-400" />
                    Block #{blocks.length - index}
                  </CardTitle>
                  <span className="text-xs text-slate-500">
                    {new Date(block.timestamp).toLocaleString()}
                  </span>
                </div>
              </CardHeader>
              <CardContent className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div>
                    <h4 className="text-sm font-semibold text-slate-300 mb-2 flex items-center">
                      <FileText className="h-4 w-4 mr-2" />
                      Detection Data
                    </h4>
                    <div className="bg-slate-900/50 rounded-lg p-3 space-y-2 text-sm">
                      <p className="text-slate-400">
                        <span className="text-slate-500">Type:</span>{' '}
                        <span className="text-white capitalize">{block.data.detection_type}</span>
                      </p>
                      {/* Camera removed from blockchain UI */}
                      <p className="text-slate-400">
                        <span className="text-slate-500">Confidence:</span>{' '}
                        <span className="text-white">{(block.data.confidence * 100).toFixed(1)}%</span>
                      </p>
                      <p className="text-slate-400">
                        <span className="text-slate-500">Location:</span>{' '}
                        <span className="text-white">
                          {block.data.location.lat.toFixed(4)}, {block.data.location.lng.toFixed(4)}
                        </span>
                      </p>
                    </div>
                  </div>

                  <div>
                    <h4 className="text-sm font-semibold text-slate-300 mb-2 flex items-center">
                      <Shield className="h-4 w-4 mr-2" />
                      Block Information
                    </h4>
                    <div className="bg-slate-900/50 rounded-lg p-3 space-y-2 text-sm">
                      <p className="text-slate-400">
                        <span className="text-slate-500">Detection ID:</span>
                      </p>
                      <code className="text-xs text-blue-400 block break-all">
                        {block.detection_id}
                      </code>
                      <p className="text-slate-400 mt-2">
                        <span className="text-slate-500">Block ID:</span>
                      </p>
                      <code className="text-xs text-purple-400 block break-all">
                        {block.block_id}
                      </code>
                    </div>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-slate-300 mb-2 flex items-center">
                    <Link2 className="h-4 w-4 mr-2" />
                    Previous Hash
                  </h4>
                  <div className="bg-slate-900/50 rounded-lg p-3">
                    <code className="text-xs text-yellow-400 break-all">
                      {block.previous_hash}
                    </code>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-slate-300 mb-2 flex items-center">
                    <Shield className="h-4 w-4 mr-2" />
                    Block Hash
                  </h4>
                  <div className="bg-gradient-to-r from-green-500/10 to-emerald-500/10 border border-green-500/30 rounded-lg p-3">
                    <code className="text-xs text-green-400 break-all font-mono">
                      {block.hash}
                    </code>
                  </div>
                </div>

                <div className="flex items-center justify-between pt-2 border-t border-slate-700">
                  <div className="flex items-center space-x-2 text-xs text-slate-500">
                    <Clock className="h-3 w-3" />
                    <span>{new Date(block.timestamp).toLocaleString()}</span>
                  </div>
                  <div className="bg-green-500/10 border border-green-500/30 rounded-full px-3 py-1">
                    <span className="text-xs text-green-400 font-semibold">Verified</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {blocks.length > 0 && (
        <Card className="bg-gradient-to-r from-purple-500/10 to-blue-500/10 border-purple-500/30">
          <CardContent className="py-6">
            <div className="flex items-center space-x-4">
              <div className="bg-gradient-to-br from-purple-500 to-blue-500 p-3 rounded-lg">
                <Shield className="h-8 w-8 text-white" />
              </div>
              <div>
                <h3 className="font-semibold text-white mb-1">Blockchain Integrity</h3>
                <p className="text-sm text-slate-300">
                  All {blocks.length} blocks are cryptographically linked and verified. Any tampering attempt will be
                  detected.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  );
};

export default Blockchain;
