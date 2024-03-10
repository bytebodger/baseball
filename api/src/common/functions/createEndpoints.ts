import type { Express, Request, Response } from 'express';
import { Endpoint } from '../enums/Endpoint.js';

export const createEndpoints = (api: Express) => {
   api.get(
      Endpoint.root,
      (_request: Request, response: Response) => response.status(200).send('Baseball AI API'),
   )
}