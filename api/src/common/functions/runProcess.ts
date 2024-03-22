import { spawn } from 'child_process';
import path from 'path';
import type { Result } from '../interfaces/Result.js';

export const runProcess = async (filePath: string): Promise<Result> => {
   const __dirname = path.resolve();
   let standardOutput = '';
   return new Promise(resolve => {
      const process = spawn(
         'node',
         [path.resolve(__dirname, filePath)],
         { shell: false },
      )
      process.stdout.on('data', data => {
         standardOutput += data;
         process.kill();
      });
      process.stderr.on('data', data => console.error(`Node ERROR: ${data}`));
      process.on('exit', () => resolve(JSON.parse(standardOutput)));
   })
}